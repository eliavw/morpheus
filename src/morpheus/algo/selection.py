import numpy as np
import warnings

from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.ensemble import *


def base_selection_algorithm(metadata, settings):
    """
    The easiest selection algorithm.
    """
    nb_atts = metadata["nb_atts"]
    nb_tgt = settings.get("param", 1)
    nb_iterations = settings.get("its", 1)

    nb_tgt = _set_nb_out_params_(nb_tgt, nb_atts)

    att_idx = np.array(range(nb_atts))
    result = np.zeros((1, nb_atts))

    for it_idx in range(nb_iterations):
        codes = _create_init(nb_atts, nb_tgt)

        # Shuffle the results
        np.random.shuffle(att_idx)
        codes = codes[:, att_idx]

        result = np.concatenate((result, codes))

    return result[1:, :]


def random_selection_algorithm(metadata, settings, random_state=997):
    np.random.seed(random_state)

    nb_atts = metadata["nb_atts"]
    nb_tgt = settings.get("param", 1)
    nb_iterations = settings.get("its", 1)
    fraction_missing = settings.get("fraction", 0.2)

    nb_tgt = _set_nb_out_params_(nb_tgt, nb_atts)

    att_idx = np.array(range(nb_atts))
    result = np.zeros((1, nb_atts))

    for it_idx in range(nb_iterations):
        codes = _create_init(nb_atts, nb_tgt)
        codes = _add_missing(codes, fraction=fraction_missing)
        codes = _ensure_desc_atts(codes)

        # Shuffle the results
        np.random.shuffle(att_idx)
        codes = codes[:, att_idx]

        result = np.concatenate((result, codes))

    return result[1:, :]


# Helpers
def _create_init(nb_atts, nb_tgt):
    res = np.zeros((nb_atts, nb_atts))
    for k in range(nb_tgt):
        res += np.eye(nb_atts, k=k)

    return res[0::nb_tgt, :]


def _add_missing(init, fraction=0.2):
    random = np.random.rand(*init.shape)

    noise = np.where(init == 0, random, init)
    missing = np.where(noise < fraction, -1, noise)

    res = np.floor(missing)

    res = _ensure_desc_atts(res)
    return res


def _ensure_desc_atts(m_codes):
    """
    If there are no input attributes in a code, we flip one missing attribute at random.
    """
    for row in m_codes:
        if 0 not in np.unique(row):
            idx_of_minus_ones = np.where(row == -1)[0]
            idx_to_change_to_zero = np.random.choice(idx_of_minus_ones)
            row[idx_to_change_to_zero] = 0

    return m_codes


def _set_nb_out_params_(param, nb_atts):
    if (param > 0) & (param < 1):
        nb_out_atts = int(np.ceil(param * nb_atts))
    elif (param >= 1) & (param < nb_atts):
        nb_out_atts = int(param)
    else:
        msg = """
        Impossible number of output attributes per model: {}\n
        This means the value of settings['selection']['param'] was set
        incorrectly.\n
        Re-adjusted to default; one model per attribute.
        """.format(
            param
        )
        warnings.warn(msg)
        nb_out_atts = 1

    return nb_out_atts
