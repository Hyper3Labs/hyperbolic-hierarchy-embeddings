"""Train Lorentz vs Euclidean embeddings on a graph and report the metrics.

Examples
  python -m hypembed.run --dataset product-taxonomy --dims 2 5 10
  python -m hypembed.run --dataset wordnet --root mammal.n.01 --dims 2 5 10
  python -m hypembed.run --dataset edges --path links.tsv --undirected --test-frac 0.05
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import datasets, evaluate
from .train import TrainConfig, train


def load(args) -> datasets.Graph:
    if args.dataset == "product-taxonomy":
        return datasets.google_product_taxonomy(args.path, closure=not args.direct_only)
    if args.dataset == "wordnet":
        return datasets.wordnet_nouns(args.root, max_nodes=args.max_nodes, closure=not args.direct_only)
    if args.dataset == "edges":
        return datasets.edge_list(args.path, undirected=args.undirected)
    raise ValueError(args.dataset)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["product-taxonomy", "wordnet", "edges"], required=True)
    ap.add_argument("--path", help="taxonomy file or edge list")
    ap.add_argument("--root", default="mammal.n.01", help="WordNet root synset")
    ap.add_argument("--max-nodes", type=int)
    ap.add_argument("--direct-only", action="store_true", help="train on parent links, not the transitive closure")
    ap.add_argument("--undirected", action="store_true", help="edge list: add reverse edges")
    ap.add_argument("--test-frac", type=float, default=0.0, help="hold out edges for link prediction")
    ap.add_argument("--geometries", nargs="+", default=["lorentz", "euclidean"])
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--neg", type=int, default=50)
    ap.add_argument("--lr-lorentz", type=float, default=0.1)
    ap.add_argument("--lr-euclidean", type=float, default=0.1)
    ap.add_argument("--eval-nodes", type=int, default=None, help="score a random subset of nodes (default: all, as in Nickel & Kiela)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    g = load(args)
    print(f"{args.dataset}: {g.n_nodes} nodes, {len(g.edges)} training pairs, {len(g.direct_edges)} direct edges")

    train_edges, test_edges = g.edges, None
    if args.test_frac > 0:
        train_edges, test_edges = datasets.split_edges(g.edges, args.test_frac, args.seed)
        print(f"link-prediction split: {len(train_edges)} train / {len(test_edges)} test edges")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for dim in args.dims:
        for geom in args.geometries:
            cfg = TrainConfig(
                geometry=geom,
                dim=dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                n_neg=args.neg,
                lr=args.lr_lorentz if geom == "lorentz" else args.lr_euclidean,
                seed=args.seed,
                device=args.device,
                log_every=max(1, args.epochs // 5),
            )
            t0 = time.time()
            emb = train(train_edges, g.n_nodes, cfg)
            r: dict = {"train_seconds": round(time.time() - t0, 1)}
            if test_edges is None:
                r["reconstruction"] = evaluate.reconstruction(emb, g.edges, geom, max_nodes=args.eval_nodes)
                if args.dataset != "edges":
                    r["direct_parent"] = evaluate.reconstruction(emb, g.direct_edges, geom, max_nodes=args.eval_nodes)
                    r["spearman_depth_radius"] = evaluate.depth_radius_spearman(emb, g.depths, geom)
            else:
                r["link_prediction"] = evaluate.link_prediction(emb, test_edges, geom)
            key = f"{geom}_{dim}d"
            results[key] = r
            print(f"{key}: {json.dumps(r)}")
            torch.save({"names": g.names, "embeddings": emb.cpu(), "geometry": geom}, out / f"{args.dataset}_{key}.pt")

    (out / f"{args.dataset}_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n{'model':16s} " + "  ".join(["MAP", "MR", "parent MAP"] if test_edges is None else ["MRR", "Hits@10"]))
    for key, r in results.items():
        if test_edges is None:
            rec, par = r["reconstruction"], r.get("direct_parent", {})
            print(f"{key:16s} {rec['map']:.3f}  {rec['mean_rank']:8.2f}  {par.get('map', float('nan')):.3f}")
        else:
            lp = r["link_prediction"]
            print(f"{key:16s} {lp['mrr']:.3f}  {lp['hits@10']:.3f}")


if __name__ == "__main__":
    main()
