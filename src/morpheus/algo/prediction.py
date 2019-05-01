import copy
import networkx as nx
import numpy as np

from functools import reduce
from morpheus.graph import add_imputation_nodes, add_merge_nodes, get_ids
from morpheus.utils import debug_print, code_to_query, query_to_code

VERBOSITY = 0


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
        debug_print(msg, level=1, V=VERBOSITY)

        return result

    thresholds = np.clip(np.arange(init_threshold, -stepsize, -stepsize), 0, 1)

    for thr in thresholds:
        g_relevant = [g for g in g_list if criterion(g) > thr]
        if len(g_relevant) > 0:
            msg = """
            We have selected {0} model(s) at threshold: {1:.2f}
            """.format(
                len(g_relevant), thr
            )
            debug_print(msg, level=0, V=VERBOSITY)
            break

    g_relevant = [copy.deepcopy(g) for g in g_relevant]
    g_relevant = [add_imputation_nodes(g, q_desc) for g in g_relevant]
    result = reduce(nx.compose, g_relevant)

    add_merge_nodes(result)

    return result


def mrai_algorithm(
    g_list,
    q_code,
    init_threshold=1.0,
    stepsize=0.1,
    complete=False,
    return_leftovers=False,
):
    q_desc, q_targ, q_miss = code_to_query(q_code)

    if complete:

        def stopping_criterion(list_of_graphs):
            outputs = (get_ids(g, kind="tgt") for g in list_of_graphs)
            outputs = reduce(set.union, outputs, set())

            return len(outputs.intersection(q_targ)) == len(q_targ)

    else:

        def stopping_criterion(list_of_graphs):
            return len(list_of_graphs) > 0

    def criterion(g):
        src = get_ids(g, kind='src')
        tgt = get_ids(g, kind='tgt')

        forbidden_inputs = len(set(q_targ).intersection(src))
        relevant = len(set(q_targ).intersection(tgt)) == 0

        feature_importances_available = [
            g.nodes()[node]["fi"]
            for node, in_degree in g.in_degree()
            if in_degree == 0
            if g.nodes()[node]["kind"] == "data"
            if g.nodes()[node]["idx"] in q_desc
        ]

        quantifier = np.sum(feature_importances_available)

        result = 0
        result -= 10*relevant   # Ten penalty points for not predicting something useful
        result -= 1*forbidden_inputs           # One penalty point per target as input
        result += quantifier

        msg = """
        relevant:           {}
        forbidden inputs:   {}
        quantifier:         {}
        result:             {}
        """.format(
            relevant, forbidden_inputs ,quantifier, result
        )
        debug_print(msg, level=1, V=VERBOSITY)

        return result

    thresholds = np.clip(np.arange(init_threshold, -1-stepsize, -stepsize), -1, 1)

    for thr in thresholds:
        g_relevant = [g for g in g_list if criterion(g) > thr]

        if stopping_criterion(g_relevant):
            mod_ids = [
                [n for n in g.nodes() if g.nodes()[n]["kind"] == "model"]
                for g in g_relevant
            ]

            msg = """
                        We have selected    {0} model(s) 
                        at threshold:       {1:.2f}
                        with model ids:     {2}
                        """.format(
                len(g_relevant), thr, mod_ids
            )
            debug_print(msg, level=0, V=VERBOSITY)
            break

    if return_leftovers:
        g_leftovers = [g for g in g_list if g not in g_relevant]

    g_relevant = [copy.deepcopy(g) for g in g_relevant]
    g_relevant = [add_imputation_nodes(g, q_desc) for g in g_relevant]
    result = reduce(nx.compose, g_relevant)

    add_merge_nodes(result)

    if return_leftovers:
        return result, g_leftovers
    else:
        return result


def it_algorithm(g_list, q_code, max_steps=4):
    def stopping_criterion(known_attributes, target_attributes):
        return len(set(target_attributes).difference(known_attributes)) == 0

    # Init
    q_desc, q_targ, q_miss = code_to_query(q_code)

    avl_desc = set(q_desc)
    avl_targ = set(q_targ + q_miss)

    avl_grph = g_list

    g_res = nx.DiGraph()

    for step in range(max_steps):

        last = step + 1 == max_steps  # Bool that indicates the last step

        if last:
            """
            avl_desc = whatever you know at this point.
            So, given this is the last step, whatever attribute that was present in
            the original target set, but is not yet known at this pointm is what you should
            focus on in this last step.
            """
            # avl_desc = what you know at this point.
            # Hence, whatever you do not know from the original targets, is what you shou
            avl_targ = set(q_targ).difference(avl_desc)

        msg = """
        Starting step:      {}
        Available targets:  {}
        Available desc   :  {}
        """.format(
            step, avl_targ, avl_desc
        )
        debug_print(msg, level=0, V=VERBOSITY)

        # Do things
        q_code = query_to_code(avl_desc, avl_targ, [])

        g_nxt, avl_grph = mrai_algorithm(
            avl_grph, q_code, complete=last, return_leftovers=True
        )

        g_res = nx.compose(g_res, g_nxt)

        # Prepare next step
        g_nxt_targ = get_ids(g_nxt, kind="targ")

        avl_desc = avl_desc.union(g_nxt_targ)
        avl_targ = avl_targ.difference(g_nxt_targ)

        if stopping_criterion(avl_desc, q_targ):
            break

    g_res = _prune(g_res, q_targ)

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

    msg = """
    tgt_nodes:          {}
    tgt_nodes[0]:       {}
    type(tgt_nodes[0]): {}
    """.format(
        tgt_nodes, tgt_nodes[0], type(tgt_nodes[0])
    )
    debug_print(msg, level=1, V=VERBOSITY)

    if tgt_nodes is None:
        tgt_nodes = [
            n
            for n, out_degree in g.out_degree()
            if out_degree == 0
            if g.nodes()[n]["kind"] == "data"
        ]
        print(tgt_nodes)
    elif isinstance(tgt_nodes[0], (int, np.int64)):
        tgt_nodes = [
            n
            for n in g.nodes()
            if g.nodes()[n]["kind"] == "data"
            if g.nodes()[n]["idx"] in tgt_nodes
        ]
    else:
        assert isinstance(tgt_nodes[0], str)

    ancestors = [nx.ancestors(g, source=n) for n in tgt_nodes]
    retain_nodes = reduce(set.union, ancestors, set(tgt_nodes))

    nodes_to_remove = [n for n in g.nodes() if n not in retain_nodes]
    for n in nodes_to_remove:
        g.remove_node(n)

    return g
