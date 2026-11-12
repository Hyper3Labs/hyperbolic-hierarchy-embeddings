"""Lorentz (hyperbolic) vs Euclidean shallow embeddings of hierarchies and graphs."""

from .datasets import Graph, edge_list, google_product_taxonomy, split_edges, wordnet_nouns
from .train import TrainConfig, train

__all__ = [
    "Graph",
    "TrainConfig",
    "edge_list",
    "google_product_taxonomy",
    "split_edges",
    "train",
    "wordnet_nouns",
]
