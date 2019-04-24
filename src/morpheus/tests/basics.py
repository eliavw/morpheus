import pandas as pd

from morpheus import SequentialComposition, ParallelComposition

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def default_dataset(random_state=997):
    """
    Generate a dataset to be used in tests.

    Returns:

    """
    X, y = make_classification(
        n_samples=10 ** 3,
        n_features=7,
        n_informative=7,
        n_repeated=0,
        n_redundant=0,
        n_clusters_per_class=2,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    train = pd.DataFrame(X_train)
    train = train.assign(y=y_train)

    test = pd.DataFrame(X_test)
    test = test.assign(y=y_test)

    return train, test


def default_chain(random_state=997):
    """
    Default classifier chain.

    For use in further tests.

    Returns:

    """
    train, _ = default_dataset(random_state=997)

    m_list = default_m_list_for_chain(train.values)

    sc = SequentialComposition()
    for m in m_list:
        sc.add_estimator(m, location="back")

    return sc


def default_ensemble(random_state=997):
    """
    Default classifier ensmeble.

    For use in further tests.

    Returns:

    """
    train, _ = default_dataset(random_state=random_state)

    m_list = default_m_list_for_ensemble(train.values)

    pc = ParallelComposition()
    for m in m_list:
        pc.add_estimator(m)

    return pc


def default_m_list_for_chain(data):
    targ_ids_1 = [4, 5]
    desc_ids_1 = [0, 1, 2]

    targ_ids_2 = [7]
    desc_ids_2 = [1, 2, 5]

    all_desc_ids = [desc_ids_1, desc_ids_2]
    all_targ_ids = [targ_ids_1, targ_ids_2]

    m_list = []
    ids = zip(all_desc_ids, all_targ_ids)

    for desc_ids, targ_ids in ids:
        msg = """
        Learning model with desc ids:    {}
                            targ ids:    {}
        """.format(
            desc_ids, targ_ids
        )
        print(msg)

        if set(targ_ids).issubset({6, 7}):
            learner = RandomForestClassifier
        elif set(targ_ids).issubset({0, 1, 2, 3, 4, 5}):
            learner = RandomForestRegressor
        else:
            msg = """
            Cannot learn mixed (nominal/numeric) models
            """
            raise ValueError(msg)

        # Learn a model for desc_ids-targ_ids
        m = learn_model(data, desc_ids, targ_ids, learner, max_depth=5, n_estimators=5)
        m_list.append(m)

    return m_list


def default_m_list_for_ensemble(data):
    targ_ids_1 = [5]
    desc_ids_1 = [0, 1, 2]

    targ_ids_2 = [4, 5]
    desc_ids_2 = [0, 1, 3]

    all_desc_ids = [desc_ids_1, desc_ids_2]
    all_targ_ids = [targ_ids_1, targ_ids_2]

    m_list = []
    ids = zip(all_desc_ids, all_targ_ids)

    for desc_ids, targ_ids in ids:
        msg = """
        Learning model with desc ids:    {}
                            targ ids:    {}
        """.format(
            desc_ids, targ_ids
        )
        print(msg)

        if set(targ_ids).issubset({6, 7}):
            learner = RandomForestClassifier
        elif set(targ_ids).issubset({0, 1, 2, 3, 4, 5}):
            learner = RandomForestRegressor
        else:
            msg = """
            Cannot learn mixed (nominal/numeric) models
            """
            raise ValueError(msg)

        # Learn a model for desc_ids-targ_ids
        m = learn_model(data, desc_ids, targ_ids, learner, max_depth=5, n_estimators=5)
        m_list.append(m)

    return m_list


def learn_model(data, desc_ids, targ_ids, model, **kwargs):
    """
    Learn a model from the data.

    The desc ids and targ ids identify which prediction task
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

    X, Y = data[:, desc_ids], data[:, targ_ids]

    if X.shape[1] == 1:
        X = X.ravel()
    if Y.shape[1] == 1:
        Y = Y.ravel()

    try:
        clf = model(**kwargs)
        clf.fit(X, Y)
    except ValueError as e:
        print(e)

    # Bookkeeping
    clf.desc_ids = desc_ids
    clf.targ_ids = targ_ids
    return clf
