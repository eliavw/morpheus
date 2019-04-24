import networkx as nx

from itertools import product


def model_to_graph(model, idx=0):
    """
    Convert a model to a network.

    Given a model with inputs and outputs, we convert it to a graphical
    model that represents it.

    Args:
        model:
        idx:        int, default=0
                    id of the model

    Returns:

    """

    G = nx.DiGraph()

    # Create nodes
    func_nodes = [
        (
            node_label(idx, kind="model"),
            {
                "kind": "model",
                "idx": idx,
                "mod": model,
                "function": model.predict,
                "src": model.desc_ids,
                "tgt": model.targ_ids,
            },
        )
    ]
    data_nodes_src = [
        (node_label(i, kind="data"), {"kind": "data", "idx": i, "tgt": [i]})
        for i in model.desc_ids
    ]
    data_nodes_tgt = [
        (node_label(i, kind="data"), {"kind": "data", "idx": i, "tgt": [i]})
        for i in model.targ_ids
    ]

    G.add_nodes_from(data_nodes_src, bipartite="data")
    G.add_nodes_from(data_nodes_tgt, bipartite="data")
    G.add_nodes_from(func_nodes, bipartite="func")

    # Create edges
    func_nodes = [t[0] for t in func_nodes]
    data_nodes_src = [t[0] for t in data_nodes_src]
    data_nodes_tgt = [t[0] for t in data_nodes_tgt]

    src_edges = [
        (*e, {"idx": d})
        for e, d in zip(product(data_nodes_src, func_nodes), model.desc_ids)
    ]
    tgt_edges = [
        (*e, {"idx": d})
        for e, d in zip(product(func_nodes, data_nodes_tgt), model.targ_ids)
    ]

    G.add_edges_from(src_edges)
    G.add_edges_from(tgt_edges)

    # G = add_stage(G)

    return G


def node_label(idx, kind="function"):
    """
    Generate a unique name for a node.

    Args:
        idx:        int
                    Node id
        kind:       str, {'func', 'model', 'function'} or {'data}
                    Every node represents either a function or data.

    Returns:

    """

    if kind in {"func", "model", "function"}:
        c = "f"
    elif kind in {"data"}:
        c = "d"
    else:
        msg = """
        Did not recognize kind:     {}
        """.format(
            kind
        )
        raise ValueError(msg)

    return "{}-{:02d}".format(c, idx)


def add_FI_to_graph(G):
    a, b = nx.bipartite.sets(G)
    m_nodes = a if G.nodes()[a.pop()]["bipartite"] == "func" else b

    m = m_nodes.pop()

    e_desc = G.in_edges({m})

    desc_ids = G.nodes()[m]["mod"].desc_ids
    feat_imp = G.nodes()[m]["mod"].feature_importances_

    fi = dict(zip(desc_ids, feat_imp))

    for e in e_desc:
        e_idx = G.edges()[e]["id"]
        G.edges()[e]["FI"] = fi[e_idx]

    return G


def add_stage(G):
    a, b = nx.bipartite.sets(G)
    m_nodes = a if G.nodes()[a.pop()]["bipartite"] == "func" else b

    m = m_nodes.pop()

    e_desc = G.in_edges({m})
    e_targ = G.out_edges({m})

    for n, _ in e_desc:
        G.nodes()[n]["stage"] = 0

    for n in [m]:
        G.nodes()[n]["stage"] = 1

    for _, n in e_targ:
        G.nodes()[n]["stage"] = 2

    return G


def convert_positions_to_dot_format(G):
    def position(x, y):
        return "{}, {}!".format(x, y)

    for n in G.nodes():
        x, y = G.nodes()[n]["pos"]
        G.nodes()[n]["pos"] = position(x, y)
    return G


def add_merge_nodes(G):
    relevant_nodes = [
        node
        for node, in_degree in G.in_degree()
        if in_degree > 1
        if G.nodes()[node]["bipartite"] == "data"
    ]
    for node in relevant_nodes:
        add_merge_node(G, node)
    return


def add_merge_node(G, original_node_label):
    original_node_attributes = G.nodes(data=True)[original_node_label]

    original_node = (original_node_label, original_node_attributes)

    merge_node_label = convert_data_node_to_merge_node(G, original_node_label)
    place_original_data_node_behind_merge_node(G, original_node, merge_node_label)

    return


def convert_data_node_to_merge_node(G, data_node_label):
    assert G.nodes()[data_node_label]["bipartite"] == "data"

    mapping = {}
    mapping[data_node_label] = "M({})".format(data_node_label)
    merge_node_label = mapping[data_node_label]

    nx.relabel_nodes(G, mapping, copy=False)

    G.nodes()[merge_node_label]["shape"] = '"triangle"'
    G.nodes()[merge_node_label]["kind"] = "merge"
    # G.nodes()[merge_node_label]["function"] = np.mean

    return mapping[data_node_label]


def place_original_data_node_behind_merge_node(G, original_node, merge_node_label):
    G.add_node(original_node[0], **original_node[1])
    G.add_edge(merge_node_label, original_node[0], idx=original_node[1]["idx"])

    return G
