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
    func_nodes = [(node_label(idx, kind="function"), {"idx": idx, "mod": model})]
    data_nodes_src = [(node_label(i, kind="data"), {"idx": i}) for i in model.desc_ids]
    data_nodes_tgt = [(node_label(i, kind="data"), {"idx": i}) for i in model.targ_ids]

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
        raise ValueError

    return "{}-{:04d}".format(c, idx)


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


def add_positions(G):
    a, b = nx.bipartite.sets(G)
    m_nodes = a if G.nodes()[a.pop()]["bipartite"] == "func" else b

    m = m_nodes.pop()

    e_desc = G.in_edges({m})
    e_targ = G.out_edges({m})

    for n, _ in e_desc:
        G.nodes()[n]["pos"] = (0, G.nodes()[n]["idx"])

    for n in [m]:
        G.nodes()[n]["pos"] = (1, 0)

    for _, n in e_targ:
        G.nodes()[n]["pos"] = (2, G.nodes()[n]["idx"])

    return G


def convert_positions_to_dot_format(G):
    def position(x, y):
        return "{}, {}!".format(x, y)

    for n in G.nodes():
        x, y = G.nodes()[n]["pos"]
        G.nodes()[n]["pos"] = position(x, y)
    return G
