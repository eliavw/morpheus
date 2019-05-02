import networkx as nx
import numpy as np

from functools import partial

from morpheus.composition import o, x


def base_inference_algorithm(g):
    # Convert the graph to its functions
    sorted_list = list(nx.topological_sort(g))
    print(sorted_list)
    functions = {}

    for node_name in sorted_list:
        node = g.nodes(data=True)[node_name]
        print(node_name)
        if node.get("kind", None) == "data":
            if len(nx.ancestors(g, node_name)) == 0:
                functions[node_name] = _select(node["idx"])
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

        elif node.get("kind", None) == "model":

            previous_nodes = [t[0] for t in g.in_edges(node_name)]

            print(functions.keys())
            inputs = [functions[n] for n in previous_nodes]
            inputs = o(np.transpose, x(*inputs, return_type=np.array))
            f = node["function"]
            functions[node_name] = o(f, inputs)

        elif node.get("kind", None) == "merge":
            merge_idx = node["idx"]
            previous_nodes = [t[0] for t in g.in_edges(node_name)]
            previous_t_idx = [g.nodes()[n]["tgt"] for n in previous_nodes]
            print(previous_t_idx)
            inputs = [(functions[n], t) for n, t in zip(previous_nodes, previous_t_idx)]

            inputs = [o(_select(t_idx.index(merge_idx)), f) for f, t_idx in inputs]
            inputs = o(np.transpose, x(*inputs, return_type=np.array))

            f = partial(np.mean, axis=1)
            functions[node_name] = o(f, inputs)

    return functions


def _select(idx):
    def select(X):
        if len(X.shape) > 1:
            return X[:, idx]
        elif len(X.shape) == 1:
            assert idx == 0
            return X

    return select
