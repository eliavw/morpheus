import numpy as np

from inspect import signature

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import morpheus.algo.selection as selection
import morpheus.algo.prediction as prediction
import morpheus.algo.inference as inference

from morpheus.algo.induction import base_induction_algorithm
from morpheus.composition import o, x
from morpheus.graph import model_to_graph


class Morpheus(object):

    # TODO: Determine these automatically based on method names.
    delimiter = "_"

    selection_algorithms = {
        "base": selection.base_selection_algorithm,
        "random": selection.random_selection_algorithm,
    }

    classifier_algorithms = {"DT": DecisionTreeClassifier, "RF": RandomForestClassifier}

    regressor_algorithms = {"DT": DecisionTreeRegressor, "RF": RandomForestRegressor}

    prediction_algorithms = {
        "mi": prediction.mi_algorithm,
        "ma": prediction.ma_algorithm,
        "mrai": prediction.mrai_algorithm,
        "it": prediction.it_algorithm,
        "rw": prediction.rw_algorithm,
    }

    inference_algorithms = {"base": inference.base_inference_algorithm}

    configuration_prefixes = {
        "selection": ["selection", "sel", "s"],
        "prediction": ["prediction", "pred", "prd", "p"],
        "inference": ["inference", "infr", "inf"],
        "classification": ["classification", "classifier", "clf", "c"],
        "regression": ["regression", "regressor", "rgr", "r"],
    }

    def __init__(
        self,
        selection_algorithm="base",
        classifier_algorithm="DT",
        regressor_algorithm="DT",
        prediction_algorithm="mi",
        inference_algorithm="base",
        random_state=997,
        **kwargs,
    ):
        self.random_state = random_state
        self.selection_algorithm = self.selection_algorithms[selection_algorithm]
        self.classifier_algorithm = self.classifier_algorithms[classifier_algorithm]
        self.regressor_algorithm = self.regressor_algorithms[regressor_algorithm]
        self.prediction_algorithm = self.prediction_algorithms[prediction_algorithm]
        self.inference_algorithm = self.inference_algorithms[inference_algorithm]
        self.induction_algorithm = (
            base_induction_algorithm
        )  # For now, we only have one.

        self.m_codes = np.array([])
        self.m_list = []
        self.g_list = []
        self.i_list = []

        self.q_diagram = None
        self.q_methods = []

        # Configurations
        self.sel_cfg = self.default_config(self.selection_algorithm)
        self.clf_cfg = self.default_config(self.classifier_algorithm)
        self.rgr_cfg = self.default_config(self.regressor_algorithm)
        self.prd_cfg = self.default_config(self.prediction_algorithm)
        self.inf_cfg = self.default_config(self.inference_algorithm)

        self.configuration = dict(
            selection=self.sel_cfg,
            classification=self.clf_cfg,
            regression=self.rgr_cfg,
            prediction=self.prd_cfg,
            inference=self.inf_cfg,
        )

        self.update_config(random_state=random_state, **kwargs)

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

        self.metadata = self.default_metadata(X)
        self._fit_imputer(X)

        self.m_codes = self.selection_algorithm(
            self.metadata, **self.sel_cfg, random_state=self.random_state
        )

        self.m_list = self.induction_algorithm(
            X,
            self.m_codes,
            classifier=self.classifier_algorithm,
            regressor=self.regressor_algorithm,
            classifier_kwargs=self.clf_cfg,
            regressor_kwargs=self.rgr_cfg,
        )
        self.g_list = [model_to_graph(m, idx) for idx, m in enumerate(self.m_list)]

        return

    def predict(self, X, q_code):

        # Make custom diagram
        self.q_diagram = self.prediction_algorithm(
            self.g_list, q_code, **self.prd_cfg, random_state=self.random_state
        )
        self.q_diagram = self._add_imputer_function(self.q_diagram)

        # Convert diagram to methods
        self.q_methods = self.inference_algorithm(self.q_diagram)

        # Execute our custom function

        return

    # Configuration
    @staticmethod
    def default_config(method):
        config = {}
        sgn = signature(method)

        for key, parameter in sgn.parameters.items():
            if parameter.default is not parameter.empty:
                config[key] = parameter.default
        return config

    def update_config(self, **kwargs):

        for kind in self.configuration:
            # Immediate matches
            overlap = set(self.configuration[kind]).intersection(set(kwargs))

            for k in overlap:
                self.configuration[kind][k] = kwargs[k]

            # Parsed matches
            parameter_map = self._parse_kwargs(kind=kind, **kwargs)
            overlap = set(self.configuration[kind]).intersection(set(parameter_map))

            for k in overlap:
                self.configuration[kind][k] = kwargs[parameter_map[k]]

        return

    def _parse_kwargs(self, kind="selection", **kwargs):

        prefixes = [e + self.delimiter for e in self.configuration_prefixes[kind]]

        parameter_map = {
            x.split(prefix)[1]: x
            for x in kwargs
            for prefix in prefixes
            if x.startswith(prefix)
        }

        return parameter_map

    def default_metadata(self, X):
        if X.ndim != 2:
            X = X.reshape(-1, 1)
        n_rows, n_cols = X.shape

        types = [X[0, 0].dtype for _ in range(n_cols)]
        nominal_attributes = [self._is_nominal(t) for t in types]
        numeric_attributes = [self._is_numeric(t) for t in types]

        metadata = dict(
            n_attributes=n_cols,
            types=types,
            nominal_attributes=nominal_attributes,
            numeric_attributes=numeric_attributes,
        )
        return metadata

    @staticmethod
    def _is_nominal(t):
        condition_01 = t == np.dtype(int)
        return condition_01

    @staticmethod
    def _is_numeric(t):
        condition_01 = t == np.dtype(float)
        return condition_01

    # Imputer
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
