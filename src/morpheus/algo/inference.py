import networkx as nx
import numpy as np

from functools import partial

from morpheus.graph.network import get_nodes, get_ids, node_label
from morpheus.composition import o, x

from morpheus.utils import debug_print

VERBOSITY = 1


def base_inference_algorithm(g, q_desc_ids=None):

    # Convert the graph to its functions
    sorted_list = list(nx.topological_sort(g))

    msg = """
    sorted_list:    {}
    """.format(
        sorted_list
    )
    debug_print(msg, level=1, V=VERBOSITY)
    functions = {}

    if q_desc_ids is None:
        # desc ids not provided => all attributes which are diagrammatically identified as descriptive, are assumed
        # to be given as inputs
        q_desc_ids = list(get_ids(g, kind="desc"))
        #q_desc_ids.sort()
        print(q_desc_ids)

    for node_name in sorted_list:
        node = g.nodes(data=True)[node_name]

        if node.get("kind", None) == "data":
            if len(nx.ancestors(g, node_name)) == 0:
                functions[node_name] = _select(q_desc_ids.index(node["idx"]))
            else:
                # This is pretty much identical to what happens in the merge node
                previous_node = [t[0] for t in g.in_edges(node_name)][
                    0
                ]  # I know it is just one
                previous_t_idx = g.nodes()[previous_node]["tgt"]

                relevant_idx = previous_t_idx.index(node["idx"])

                functions[node_name] = o(
                    _select(relevant_idx), functions[previous_node]
                )

        elif node.get("kind", None) == "imputation":
            functions[node_name] = node["function"]

        elif node.get("kind", None) == "model":
            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            inputs = {g.nodes()[n]["tgt"][0]: functions[n] for n in previous_nodes}
            inputs = [inputs[k] for k in sorted(inputs)] # We need to sort to get the inputs in the correct order.

            inputs = o(np.transpose, x(*inputs, return_type=np.array))
            f = node["function"]
            functions[node_name] = o(f, inputs)

        elif node.get("kind", None) == "merge":
            merge_idx = node["idx"]
            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            previous_t_idx = [g.nodes()[n]["tgt"] for n in previous_nodes]

            inputs = [(functions[n], t) for n, t in zip(previous_nodes, previous_t_idx)]

            inputs = [o(_select(t_idx.index(merge_idx)), f) for f, t_idx in inputs]
            inputs = o(np.transpose, x(*inputs, return_type=np.array))

            f = partial(np.mean, axis=1)
            functions[node_name] = o(f, inputs)

    return functions


def get_predict(methods, q_targ_ids):
    """
    Compose single predict function for a diagram.

    Parameters
    ----------
    diagram
    methods

    Returns
    -------

    """
    q_targ_ids.sort()

    tgt_methods = [methods[node_label(t, kind="data")] for t in q_targ_ids]

    return o(np.transpose, x(*tgt_methods, return_type=np.array))


def _select(idx):
    def select(X):
        if len(X.shape) > 1:
            return X[:, idx]
        elif len(X.shape) == 1:
            assert idx == 0
            return X

    return select
