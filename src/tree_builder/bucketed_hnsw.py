from .bucketed import BucketedAbstractTreeBuilder
from .ann import AbstractTreeBuilderHNSW


class AbstractTreeBuilderBucketedHNSW(BucketedAbstractTreeBuilder):

    def __init__(self, conf) -> None:
        super().__init__(conf)
        self._ann_builder = AbstractTreeBuilderHNSW(conf)

    def _get_unionfind_tree_fragment(self, node_embeddings):
        return self._ann_builder._get_unionfind_tree(node_embeddings)
