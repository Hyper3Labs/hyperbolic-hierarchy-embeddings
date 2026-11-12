# Hyperbolic hierarchy embeddings

Small, readable PyTorch code for training **hyperbolic (Lorentz model)** and **Euclidean** shallow embeddings on a graph, with the same loss, sampler and optimizer for both, so geometry is the only variable.

It covers the two hierarchies behind our posts:

- **WordNet nouns**: the Nickel & Kiela benchmark used in [The Geometry Mistake Behind Modern Embedding Models](https://hyper3labs.com/blog/the-geometry-mistake/).
- **Google Product Taxonomy**: 5,595 categories, the dataset in [Hyperbolic Embeddings in Qdrant](https://qdrant.tech/articles/hyperbolic-embeddings-qdrant/).

It also takes **any edge list**, which is the entry point for non-hierarchical graphs such as a web host graph.

## Quick start

```bash
pip install -e ".[wordnet]"     # or: uv sync --extra wordnet

# Google Product Taxonomy (downloads the taxonomy file from Google on first run)
python -m hypembed.run --dataset product-taxonomy --dims 2 5 10

# WordNet mammal subtree, or the full noun hierarchy with --root entity.n.01
python -m hypembed.run --dataset wordnet --root mammal.n.01 --dims 2 5 10
python -m hypembed.run --dataset wordnet --root entity.n.01 --dims 5 10   # ~3 h per model on one CPU core

# Any graph: one "src dst" per line, held-out edges for link prediction
python -m hypembed.run --dataset edges --path links.tsv --undirected --test-frac 0.05
```

Each run writes `outputs/<dataset>_<geometry>_<dim>d.pt` (node names + embeddings) and `outputs/<dataset>_results.json`.

## What's in it

| File | Contents |
|---|---|
| `src/hypembed/lorentz.py` | Lorentz inner product, distance, exp map, Riemannian gradient, projection, map to the Poincaré ball |
| `src/hypembed/train.py` | One trainer for both geometries: softmax over 1 positive + K negatives on −distance, sparse row updates, Riemannian SGD (exp map) for Lorentz, burn-in at lr/10 |
| `src/hypembed/evaluate.py` | Reconstruction mean rank / MAP (Nickel & Kiela), link-prediction MRR / Hits@k against random negatives, depth↔radius Spearman |
| `src/hypembed/datasets.py` | WordNet (via nltk), Google Product Taxonomy, generic edge list, edge split that keeps (u,v) and (v,u) on the same side |
| `src/hypembed/run.py` | CLI that sweeps geometries × dimensions and prints a table |

## Results

Defaults throughout (300 epochs, batch 128, 50 negatives, lr 0.1, seed 0), trained on the transitive closure and scored on every node, on a 4-core CPU. "Parent MAP" ranks only the direct parent, which is a harder test.

**Google Product Taxonomy** (5,595 nodes, 17,312 ancestor pairs)

| Model | MAP | Mean rank | Parent MAP | Qdrant article MAP |
|---|---:|---:|---:|---:|
| Lorentz 2D | 0.541 | 3.90 | 0.504 | 0.501 |
| Euclidean 2D | 0.154 | 75.11 | 0.274 | 0.140 |
| Lorentz 5D | **0.903** | **1.41** | 0.663 | 0.905 |
| Euclidean 5D | 0.240 | 40.42 | 0.469 | 0.239 |
| Lorentz 10D | **0.912** | **1.32** | 0.660 | 0.925 |
| Euclidean 10D | 0.353 | 16.38 | 0.647 | 0.354 |

The article's numbers come from a Poincaré-ball trainer with the same loss and batch size. Training longer helps further: with `--epochs 600`, Lorentz reaches 0.948 (5D) and 0.954 (10D).

**WordNet nouns** (`--root entity.n.01`, 82,115 nodes, 743,241 closure pairs, the Nickel & Kiela benchmark)

| Model | MAP | Mean rank | Parent MAP | Nickel & Kiela 2017 MAP / MR |
|---|---:|---:|---:|---:|
| Lorentz 5D | 0.735 | 32.0 | 0.139 | 0.823 / 4.9 (Poincaré) |
| Euclidean 5D | 0.068 | 3762.3 | 0.241 | 0.024 / 3542.3 |
| Lorentz 10D | 0.760 | 24.6 | 0.125 | 0.851 / 4.02 (Poincaré) |
| Euclidean 10D | 0.084 | 1539.6 | 0.283 | 0.059 / 2286.9 |

About 2.5 to 3 hours per hyperbolic model on one CPU core. On full WordNet the hyperbolic models land about 0.09 MAP below Nickel & Kiela, with a higher mean rank, while still scoring about ten times the Euclidean MAP. Training to 600 epochs only moves MAP to 0.759 (5D) and 0.763 (10D), and a higher learning rate (0.3 or 1.0) makes it worse (0.57 and 0.31 at 5D), so closing the rest would need Nickel & Kiela's burn-in and much longer schedule rather than a different default.

**WordNet mammals** (`mammal.n.01`, 1,182 nodes, 6,542 closure pairs)

| Model | MAP | Mean rank | Parent MAP |
|---|---:|---:|---:|
| Lorentz 2D | 0.844 | 2.58 | 0.306 |
| Euclidean 2D | 0.145 | 140.89 | 0.308 |
| Lorentz 5D | **0.940** | **1.63** | 0.253 |
| Euclidean 5D | 0.236 | 81.26 | 0.576 |
| Lorentz 10D | **0.940** | **1.49** | 0.237 |
| Euclidean 10D | 0.397 | 31.29 | 0.767 |

Hyperbolic wins clearly on reconstructing the hierarchy at low dimension. It does not win on recovering the direct parent from closure training. These are reconstruction scores on seen relations, not generalization to held-out edges. For held-out edges, use `--test-frac`.

## References

- M. Nickel, D. Kiela. *Poincaré Embeddings for Learning Hierarchical Representations.* NeurIPS 2017.
- M. Nickel, D. Kiela. *Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry.* ICML 2018.
- S. Bansal, A. Benton. *Comparing Euclidean and Hyperbolic Embeddings on the WordNet Nouns Hypernymy Graph.* Insights from Negative Results in NLP, 2021.

## License

MIT. The Google Product Taxonomy and WordNet are downloaded from their sources at run time and are not redistributed here.
