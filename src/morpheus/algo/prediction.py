import copy
import networkx as nx
import numpy as np

from functools import reduce
from morpheus.graph import add_imputation_nodes, add_merge_nodes, get_ids
from morpheus.utils.encoding import code_to_query, query_to_code


def mi_algorithm(g_list, q_code):
    q_desc, q_targ, q_miss = code_to_query(q_code)

    def criterion(g):
        outputs = set(
            [
                g.nodes()[node]["idx"]
                for node, out_degree in g.out_degree()
                if out_degree == 0
                if g.nodes()[node]["kind"] == "data"
            ]
        )

        return len(set(q_targ).intersection(outputs)) > 0

    g_relevant = [g for g in g_list if criterion(g)]
    g_relevant = [copy.copy(g) for g in g_relevant]

    g_relevant = [add_imputation_nodes(g, q_desc) for g in g_relevant]

    result = reduce(nx.compose, g_relevant)

    return result


def ma_algorithm(g_list, q_code, init_threshold=1.0, stepsize=0.1):
    q_desc, q_targ, q_miss = code_to_query(q_code)

    def criterion(g):
        inputs = set(
            [
                g.nodes()[node]["idx"]
                for node, in_degree in g.in_degree()
                if in_degree == 0
                if g.nodes()[node]["kind"] == "data"
            ]
        )

        outputs = set(
            [
                g.nodes()[node]["idx"]
                for node, out_degree in g.out_degree()
                if out_degree == 0
                if g.nodes()[node]["kind"] == "data"
            ]
        )

        yes_no = len(set(q_targ).intersection(outputs)) > 0

        quantifier = len(set(q_desc).intersection(inputs)) / len(inputs)

        result = int(yes_no) * quantifier

        msg = """
        yes_no:       {}
        quantifier:   {}
        result:       {}
        """.format(
            yes_no, quantifier, result
        )
        print(msg)

        return result

    thresholds = np.clip(np.arange(init_threshold, -stepsize, -stepsize), 0, 1)

    for thr in thresholds:
        g_relevant = [g for g in g_list if criterion(g) > thr]
        if len(g_relevant) > 0:
            print("we have found a model at threshold: {}".format(thr))
            break

    g_relevant = [copy.deepcopy(g) for g in g_relevant]
    g_relevant = [add_imputation_nodes(g, q_desc) for g in g_relevant]
    result = reduce(nx.compose, g_relevant)

    add_merge_nodes(result)

    return result


def mrai_algorithm(g_list, q_code, init_threshold=1.0, stepsize=0.1):
    q_desc, q_targ, q_miss = code_to_query(q_code)

    def criterion(g):
        outputs = set(
            [
                g.nodes()[node]["idx"]
                for node, out_degree in g.out_degree()
                if out_degree == 0
                if g.nodes()[node]["kind"] == "data"
            ]
        )

        yes_no = len(set(q_targ).intersection(outputs)) > 0

        feature_importances_available = [
            g.nodes()[node]["fi"]
            for node, in_degree in g.in_degree()
            if in_degree == 0
            if g.nodes()[node]["kind"] == "data"
            if g.nodes()[node]["idx"] in q_desc
        ]

        quantifier = np.sum(feature_importances_available)

        result = int(yes_no) * quantifier

        msg = """
        yes_no:       {}
        quantifier:   {}
        result:       {}
        """.format(
            yes_no, quantifier, result
        )
        print(msg)

        return result

    thresholds = np.clip(np.arange(init_threshold, -stepsize, -stepsize), 0, 1)

    for thr in thresholds:
        g_relevant = [g for g in g_list if criterion(g) > thr]
        if len(g_relevant) > 0:
            print("we have found a model at threshold: {}".format(thr))
            break

    g_relevant = [copy.deepcopy(g) for g in g_relevant]
    g_relevant = [add_imputation_nodes(g, q_desc) for g in g_relevant]
    result = reduce(nx.compose, g_relevant)

    add_merge_nodes(result)

    return result


def it_algorithm(g_list, q_code, max_steps=4):
    # Init
    q_desc, q_targ, q_miss = code_to_query(q_code)

    avl_desc = set(q_desc)
    avl_targ = set(q_targ + q_miss)

    avl_grph = g_list

    g_res = nx.DiGraph()

    for step in range(max_steps):
        q_code = query_to_code(avl_desc, avl_targ, [])

        g_nxt = mrai_algorithm(avl_grph, q_code, complete=(max_steps - step == 1))
        g_res = nx.compose(g_res, g_nxt)

        g_nxt_targ = get_ids(g_nxt, kind="targ")

        g_nxt_mods = set([n for n in g.nodes() if g.nodes()[n]["kind"] == "model"])

        avl_desc = avl_desc.union(g_nxt_targ)
        avl_targ = avl_targ.difference(g_nxt_targ)

        avl_grph = [g for g in avl_grph if len(g_nxt_mods.intersection(set(g))) > 0]

    g_res = _prune(g_res)

    return g_res


def rw_algorithm(g_list, q_code, max_chain_size=5):
    # Init
    q_desc, q_targ, q_miss = code_to_query(q_code)

    avl_desc = set(q_desc)
    avl_targ = set(q_targ)

    avl_grph = g_list

    g_res = nx.DiGraph()

    for step in range(max_chain_size):
        q_code = query_to_code(avl_desc, avl_targ, [])

        g_nxt = mrai_algorithm(avl_grph, q_code, stochastic=True)
        g_res = nx.compose(g_res, g_nxt)

        g_nxt_mods = set(
            [n for n in g_nxt.nodes() if g_nxt.nodes()[n]["kind"] == "model"]
        )
        g_res_desc, g_res_targ = (
            get_ids(g_res, kind="desc"),
            get_ids(g_res, kind="targ"),
        )

        avl_desc = avl_desc
        avl_targ = g_res_desc.difference(avl_desc).union(
            avl_targ.difference(g_res_targ)
        )

        avl_grph = [g for g in avl_grph if len(g_nxt_mods.intersection(set(g))) > 0]

    g_res = _prune(g_res)

    return g_res


def _prune(g, tgt_nodes=None):

    if tgt_nodes is None:
        tgt_nodes = [
            n
            for n, out_degree in g.out_degree()
            if out_degree == 0
            if g.nodes()[n]["kind"] == "data"
        ]
        print(tgt_nodes)
    else:
        assert isinstance(tgt_nodes, list)

    ancestors = [nx.ancestors(g, source=n) for n in tgt_nodes]
    ancestors = reduce(set.union, ancestors)

    nodes_to_remove = [n for n in g.nodes() if n not in ancestors]
    for n in nodes_to_remove:
        g.remove_node(n)

    return g
