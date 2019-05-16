import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from morpheus.utils import code_to_query

from morpheus.utils import debug_print

VERBOSITY = 0


def base_induction_algorithm(data, m_codes):

    # Init
    assert isinstance(data, np.ndarray)
    n_rows, n_cols = data.shape
    attributes = list(range(n_cols))
    m_list = []
    all_desc_ids, all_targ_ids = [], []

    # Codes to queries
    for m_code in m_codes:
        desc_ids, targ_ids, _ = code_to_query(m_code)
        all_desc_ids.append(desc_ids)
        all_targ_ids.append(targ_ids)

    ids = zip(all_desc_ids, all_targ_ids)
    for desc_ids, targ_ids in ids:
        msg = """
        Learning model with desc ids:    {}
                            targ ids:    {}
        """.format(
            desc_ids, targ_ids
        )
        debug_print(msg, level=1, V=VERBOSITY)

        if set(targ_ids).issubset(attributes[-1:]):
            learner = RandomForestClassifier
        elif set(targ_ids).issubset(attributes[:-1]):
            learner = RandomForestRegressor
        else:
            msg = """
            Cannot learn mixed (nominal/numeric) models
            """
            raise ValueError(msg)

        # Learn a model for desc_ids-targ_ids
        n_cols = _learn_model(
            data, desc_ids, targ_ids, learner, max_depth=5, n_estimators=5
        )
        m_list.append(n_cols)

    return m_list


def _learn_model(data, desc_ids, targ_ids, model, **kwargs):
    """
    Learn a model from the data.

    The desc ids and targ ids identify which algo task
    you should try to learn from the data.

    Model is a machine learning method that has a .fit() method.

    Args:
        data:
        desc_ids:
        targ_ids:
        model:
        **kwargs:

    Returns:

    """

    i, o = data[:, desc_ids], data[:, targ_ids]

    if i.shape[1] == 1:
        i = i.ravel()
    if o.shape[1] == 1:
        o = o.ravel()

    try:
        clf = model(**kwargs)
        clf.fit(i, o)
    except ValueError as e:
        print(e)

    # Bookkeeping
    clf.desc_ids = desc_ids
    clf.targ_ids = targ_ids
    return clf
