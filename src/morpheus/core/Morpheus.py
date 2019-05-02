import numpy as np

import morpheus.algo.selection as selection
import morpheus.algo.induction as induction
import morpheus.algo.prediction as prediction
import morpheus.algo.inference as inference


from morpheus.graph import model_to_graph


class Morpheus(object):

    # TODO: Determine these automatically based on method names.
    selection_algorithms = {
        "base": selection.base_selection_algorithm,
        "random": selection.random_selection_algorithm,
    }

    induction_algorithms = {"base": induction.base_induction_algorithm}

    prediction_algorithms = {
        "mi": prediction.mi_algorithm,
        "ma": prediction.ma_algorithm,
        "mrai": prediction.mrai_algorithm,
        "it": prediction.it_algorithm,
        "rw": prediction.rw_algorithm,
    }

    inference_algorithms = {"base": inference.base_inference_algorithm}

    def __init__(
        self,
        sel_algo="base",
        ind_algo="base",
        pred_algo="mi",
        inf_algo="base",
        random_state=997,
    ):
        self.random_state = random_state
        self.selection_algorithm = self.selection_algorithms[sel_algo]
        self.induction_algorithm = self.induction_algorithms[ind_algo]
        self.prediction_algorithm = self.prediction_algorithms[pred_algo]
        self.inference_algorithm = self.inference_algorithms[inf_algo]

        self.m_codes = np.array([])
        self.m_list = []
        self.g_list = []
        return

    def fit(self, X):
        assert isinstance(X, np.ndarray)
        n, m = X.shape

        metadata = {"nb_atts": m}
        settings = {"param": 1, "its": 1}

        self.m_codes = self.selection_algorithm(metadata, settings)
        self.m_list = self.induction_algorithm(X, self.m_codes)
        self.g_list = [model_to_graph(m, idx) for idx, m in enumerate(self.m_list)]

        return

    def predict(self, X, q_code):

        self.q_grph = self.prediction_algorithm(self.g_list, q_code)
        self.f_list = self.inference_algorithm(self.q_grph)

        return self.f_list
