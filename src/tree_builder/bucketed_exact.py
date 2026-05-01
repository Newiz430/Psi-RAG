from .bucketed import BucketedAbstractTreeBuilder


class AbstractTreeBuilderBucketedExact(BucketedAbstractTreeBuilder):

    def _get_unionfind_tree_fragment(self, node_embeddings):
        return self._get_unionfind_tree(node_embeddings)
