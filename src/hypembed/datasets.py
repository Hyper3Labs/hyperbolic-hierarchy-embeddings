"""Graphs to embed.

Every loader returns a Graph: node names, a depth per node (-1 if unknown), and
directed edges as an (E, 2) LongTensor of (child, ancestor) or (src, dst) pairs.
"""
from __future__ import annotations

import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

GOOGLE_TAXONOMY_URL = "https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Graph:
    names: list[str]
    depths: list[int]
    edges: torch.Tensor  # training / reconstruction pairs
    direct_edges: torch.Tensor  # parent links only (for plots and harder eval)

    @property
    def n_nodes(self) -> int:
        return len(self.names)


def _closure(n: int, parent_of: list[list[int]]) -> list[tuple[int, int]]:
    """(node, ancestor) pairs of a DAG given each node's direct parents."""
    memo: dict[int, set[int]] = {}

    def ancestors(i: int) -> set[int]:
        if i in memo:
            return memo[i]
        out: set[int] = set()
        stack = list(parent_of[i])
        while stack:
            p = stack.pop()
            if p not in out:
                out.add(p)
                stack.extend(parent_of[p])
        memo[i] = out
        return out

    return [(i, a) for i in range(n) for a in ancestors(i)]


def _graph(names, depths, parent_of, closure: bool) -> Graph:
    direct = [(c, p) for c, ps in enumerate(parent_of) for p in ps]
    pairs = _closure(len(names), parent_of) if closure else direct
    return Graph(
        names=names,
        depths=depths,
        edges=torch.tensor(pairs, dtype=torch.long),
        direct_edges=torch.tensor(direct, dtype=torch.long),
    )


def google_product_taxonomy(path: str | Path | None = None, *, closure: bool = True) -> Graph:
    """Google Product Taxonomy: 5,595 categories, 7 levels, 17,312 ancestor pairs."""
    path = Path(path) if path else DATA_DIR / "google-product-taxonomy.en-US.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {GOOGLE_TAXONOMY_URL} -> {path}")
        urllib.request.urlretrieve(GOOGLE_TAXONOMY_URL, path)

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            _, _, full = line.partition(" - ")
            rows.append([p.strip() for p in full.split(">")])
    index = {" > ".join(parts): i for i, parts in enumerate(rows)}
    names = [parts[-1] for parts in rows]
    depths = [len(parts) - 1 for parts in rows]
    parent_of = [[index[" > ".join(parts[:-1])]] if len(parts) > 1 else [] for parts in rows]
    return _graph(names, depths, parent_of, closure)


def wordnet_nouns(root: str = "entity.n.01", *, max_nodes: int | None = None, closure: bool = True) -> Graph:
    """WordNet noun hypernymy under `root` (e.g. mammal.n.01), via nltk.

    Full nouns (entity.n.01): ~82k synsets, ~740k closure pairs (the N&K benchmark).
    """
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        wn.ensure_loaded()
    except LookupError:
        if not nltk.download("wordnet", quiet=True):
            raise RuntimeError("could not download WordNet; run `python -m nltk.downloader wordnet`")
        wn.ensure_loaded()

    start = wn.synset(root)
    order, depth = [], {}
    q = deque([(start, 0)])
    while q and (max_nodes is None or len(order) < max_nodes):
        s, d = q.popleft()
        if s in depth:
            continue
        depth[s] = d
        order.append(s)
        # nltk returns hyponyms in set order, which changes between processes; sort for a stable node order
        for h in sorted(s.hyponyms() + s.instance_hyponyms(), key=lambda x: x.name()):
            if h not in depth:
                q.append((h, d + 1))
    idx = {s: i for i, s in enumerate(order)}
    parent_of = [
        [idx[p] for p in s.hypernyms() + s.instance_hypernyms() if p in idx] for s in order
    ]
    return _graph([s.name() for s in order], [depth[s] for s in order], parent_of, closure)


def edge_list(path: str | Path, *, sep: str | None = None, undirected: bool = False) -> Graph:
    """Any graph from a text edge list: one `src dst` pair per line (names or ids).

    This is the entry point for non-hierarchical graphs such as a host link graph.
    """
    names: dict[str, int] = {}
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            a, b = line.split(sep)[:2]
            ia = names.setdefault(a, len(names))
            ib = names.setdefault(b, len(names))
            if ia != ib:
                pairs.append((ia, ib))
                if undirected:
                    pairs.append((ib, ia))
    e = torch.tensor(pairs, dtype=torch.long)
    return Graph(names=list(names), depths=[-1] * len(names), edges=e, direct_edges=e)


def split_edges(edges: torch.Tensor, test_frac: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random edge split, grouping (u, v) and (v, u) so a reverse edge can't leak."""
    lo = torch.minimum(edges[:, 0], edges[:, 1])
    hi = torch.maximum(edges[:, 0], edges[:, 1])
    pair_id = torch.unique(torch.stack([lo, hi], 1), dim=0, return_inverse=True)[1]
    g = torch.Generator().manual_seed(seed)
    n_pairs = int(pair_id.max()) + 1
    is_test = torch.rand(n_pairs, generator=g) < test_frac
    mask = is_test[pair_id]
    return edges[~mask], edges[mask]
