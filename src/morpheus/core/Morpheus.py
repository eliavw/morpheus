import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import morpheus.algo.selection as selection
import morpheus.algo.induction as induction
import morpheus.algo.prediction as prediction
import morpheus.algo.inference as inference

from morpheus.composition import o, x
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
        selection_algorithm="base",
        induction_algorithm="base",
        prediction_algorithm="mi",
        inference_algorithm="base",
        random_state=997,
        **kwargs,
    ):
        self.random_state = random_state
        self.selection_algorithm = self.selection_algorithms[selection_algorithm]
        self.induction_algorithm = self.induction_algorithms[induction_algorithm]
        self.prediction_algorithm = self.prediction_algorithms[prediction_algorithm]
        self.inference_algorithm = self.inference_algorithms[inference_algorithm]

        self.m_codes = np.array([])
        self.m_list = []
        self.g_list = []
        self.i_list = []

        self.q_diagram = None
        self.q_methods = []

        # Configurations
        self.ind_cfg = self.update_config(mode="induction", **kwargs)
        self.sel_cfg = self.update_config(mode="selection", **kwargs)
        self.prd_cfg = self.update_config(mode="prediction", **kwargs)
        self.inf_cfg = self.update_config(mode="inference", **kwargs)

        return

    def fit(self, X):
        """
        Fit a MERCS model on data X.

        Parameters
        ----------
        X:      np.ndarray,
                training data.

        Returns
        -------

        """
        assert isinstance(X, np.ndarray)
        n_rows, n_cols = X.shape

        self._fit_imputer(X)

        metadata = {"nb_atts": n_cols}
        settings = {"param": 1, "its": 1}

        self.m_codes = self.selection_algorithm(
            metadata, settings, random_state=self.random_state
        )
        self.m_list = self.induction_algorithm(X, self.m_codes)
        self.g_list = [model_to_graph(m, idx) for idx, m in enumerate(self.m_list)]

        return

    def predict(self, X, q_code):

        # Make custom diagram
        self.q_diagram = self.prediction_algorithm(
            self.g_list, q_code, random_state=self.random_state
        )
        self.q_diagram = self._add_imputer_function(self.q_diagram)

        # Convert diagram to methods
        self.q_methods = self.inference_algorithm(self.q_diagram)

        return

    def update_config(self, mode="induction", **kwargs):

        configuration = {}

        return configuration

    def _fit_imputer(self, X):
        """
        Construct and fit an imputer
        """
        n_rows, n_cols = X.shape

        i_list = []
        for c in range(n_cols):
            i = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            i.fit(X[:, [c]])
            i_list.append(i)

        self.i_list = i_list

        return

    def _add_imputer_function(self, g):

        for n in g.nodes():
            if g.nodes()[n]["kind"] == "imputation":
                idx = g.nodes()[n]["idx"]

                f_1 = self._dummy_array  # Artificial input
                f_2 = self.i_list[idx].transform  # Actual imputation
                f_3 = np.ravel  # Return a vector, not array

                g.nodes()[n]["function"] = o(f_3, o(f_2, f_1))

        return g

    @staticmethod
    def _dummy_array(X):
        """
        Return an array of np.nan, with the same number of rows as the input array.

        Parameters
        ----------
        X:      np.ndarray(), n_rows, n_cols = X.shape,
                We use the shape of X to deduce shape of our output.

        Returns
        -------
        a:      np.ndarray(), shape= (n_rows, 1)
                n_rows is the same as the number of rows as X.

        """
        n_rows, _ = X.shape

        a = np.empty((n_rows, 1))
        a.fill(np.nan)

        return a
