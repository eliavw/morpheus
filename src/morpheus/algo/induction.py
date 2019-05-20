import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from morpheus.utils import code_to_query

from morpheus.utils import debug_print

VERBOSITY = 0


def base_induction_algorithm(
    data,
    m_codes,
    classifier=DecisionTreeClassifier,
    regressor=DecisionTreeRegressor,
    classifier_kwargs=None,
    regressor_kwargs=None,
):

    # Init
    if classifier_kwargs is None:
        classifier_kwargs = dict()
    if regressor_kwargs is None:
        regressor_kwargs = dict()

    assert isinstance(data, np.ndarray)
    n_rows, n_cols = data.shape
    attributes = list(range(n_cols))
    m_list = []
    all_desc_ids, all_targ_ids = [], []

    nominal_attributes = attributes[-1:]
    numeric_attributes = attributes[:-1]

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

        if set(targ_ids).issubset(nominal_attributes):
            kwargs = classifier_kwargs
            learner = classifier
        elif set(targ_ids).issubset(numeric_attributes):
            kwargs = regressor_kwargs
            learner = regressor
        else:
            msg = """
            Cannot learn mixed (nominal/numeric) models
            """
            raise ValueError(msg)

        # Learn a model for desc_ids-targ_ids
        n_cols = _learn_model(data, desc_ids, targ_ids, learner, **kwargs)
        m_list.append(n_cols)

    return m_list


def _learn_model(data, desc_ids, targ_ids, learner, **kwargs):
    """
    Learn a model from the data.

    The desc ids and targ ids identify which algo task
    you should try to learn from the data.

    Model is a machine learning method that has a .fit() method.

    Args:
        data:
        desc_ids:
        targ_ids:
        learner:
        **kwargs:

    Returns:

    """

    i, o = data[:, desc_ids], data[:, targ_ids]

    if i.ndim == 1:
        i = i.reshape(-1, 1)
    if o.shape[1] == 1:
        o = o.ravel()

    try:
        model = learner(**kwargs)
        model.fit(i, o)
    except ValueError as e:
        print(e)

    # Bookkeeping
    model.desc_ids = desc_ids
    model.targ_ids = targ_ids
    return model
