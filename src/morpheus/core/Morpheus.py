import numpy as np

from inspect import signature
import warnings

from networkx import NetworkXUnfeasible
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import morpheus.algo.selection as selection
import morpheus.algo.prediction as prediction
import morpheus.algo.inference as inference

from morpheus.algo.inference import get_predict
from morpheus.algo.induction import base_induction_algorithm
from morpheus.composition import o, x
from morpheus.utils.encoding import encode_attribute
from morpheus.graph import model_to_graph
from morpheus.visuals import show_diagram

DESC_ENCODING = encode_attribute(1, [1], [2])
TARG_ENCODING = encode_attribute(2, [1], [2])
MISS_ENCODING = encode_attribute(0, [1], [2])


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
        "metadata": ["metadata", "meta", "mtd", "md"],
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
        self.q_predict = None

        # Configurations
        self.sel_cfg = self._default_config(self.selection_algorithm)
        self.clf_cfg = self._default_config(self.classifier_algorithm)
        self.rgr_cfg = self._default_config(self.regressor_algorithm)
        self.prd_cfg = self._default_config(self.prediction_algorithm)
        self.inf_cfg = self._default_config(self.inference_algorithm)

        self.configuration = dict(
            selection=self.sel_cfg,
            classification=self.clf_cfg,
            regression=self.rgr_cfg,
            prediction=self.prd_cfg,
            inference=self.inf_cfg,
        )  # Collect all configs in one

        self._update_config(random_state=random_state, **kwargs)

        self.metadata = dict()

        return

    def fit(self, X, **kwargs):
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

        self.metadata = self._default_metadata(X)
        self._update_metadata(**kwargs)

        self._fit_imputer(X)

        # N.b.: We do not provide `random state` as a separate parameter, already contained in self.sel_cfg!
        self.m_codes = self.selection_algorithm(self.metadata, **self.sel_cfg)

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

    def predict(self, X, q_code=None):

        if q_code is None:
            q_code = self._default_q_code()

        # Adjust data
        q_desc_ids = np.where(q_code == DESC_ENCODING)[0].tolist()
        q_targ_ids = np.where(q_code == TARG_ENCODING)[0].tolist()
        if X.shape[1] == len(q_code):
            # Assumption: User gives array with all attributes
            X = X[:, q_desc_ids]
        else:
            # Assumption: User gives array with only descriptive attributes
            assert X.shape[1] == len(q_desc_ids)

        # Make custom diagram
        self.q_diagram = self.prediction_algorithm(self.g_list, q_code, **self.prd_cfg)
        self.q_diagram = self._add_imputer_function(self.q_diagram)

        # Convert diagram to methods.
        try:
            self.q_methods = self.inference_algorithm(
                self.q_diagram, q_desc_ids=q_desc_ids
            )
        except NetworkXUnfeasible:
            msg = """
            Topological sort failed, investigate diagram to debug.
            """
            warnings.warn(msg)

        # Custom predict function
        self.q_predict = get_predict(self.q_methods, q_targ_ids)

        return self.q_predict(X)

    def show_q_diagram(self, kind="svg", fi=False, ortho=False):
        return show_diagram(self.q_diagram, kind=kind, fi=fi, ortho=ortho)

    # Configuration
    @staticmethod
    def _default_config(method):
        config = {}
        sgn = signature(method)

        for key, parameter in sgn.parameters.items():
            if parameter.default is not parameter.empty:
                config[key] = parameter.default
        return config

    def _default_metadata(self, X):
        if X.ndim != 2:
            X = X.reshape(-1, 1)

        n_rows, n_cols = X.shape

        types = [X[0, 0].dtype for _ in range(n_cols)]
        nominal_attributes = set(
            [att for att, typ in enumerate(types) if self._is_nominal(typ)]
        )
        numeric_attributes = set(
            [att for att, typ in enumerate(types) if self._is_numeric(typ)]
        )

        metadata = dict(
            attributes=set(range(n_cols)),
            n_attributes=n_cols,
            types=types,
            nominal_attributes=nominal_attributes,
            numeric_attributes=numeric_attributes,
        )
        return metadata

    def _update_config(self, **kwargs):

        for kind in self.configuration:
            self._update_dictionary(self.configuration[kind], kind=kind, **kwargs)

        return

    def _update_metadata(self, **kwargs):

        self._update_dictionary(self.metadata, kind="metadata", **kwargs)

        # Assure every attribute is `typed`
        numeric = self.metadata["numeric_attributes"]
        nominal = self.metadata["nominal_attributes"]
        att_ids = self.metadata["attributes"]

        # If not every attribute is accounted for, set to numeric type (default)
        if len(nominal) + len(numeric) != len(att_ids):
            numeric = att_ids - nominal
            self._update_dictionary(
                self.metadata, kind="metadata", numeric_attributes=numeric
            )

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

    def _update_dictionary(self, dictionary, kind=None, **kwargs):
        # Immediate matches
        overlap = set(dictionary).intersection(set(kwargs))

        for k in overlap:
            dictionary[k] = kwargs[k]

        if kind is not None:
            # Parsed matches
            parameter_map = self._parse_kwargs(kind=kind, **kwargs)
            overlap = set(dictionary).intersection(set(parameter_map))

            for k in overlap:
                dictionary[k] = kwargs[parameter_map[k]]
        return

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

    def _default_q_code(self):

        q_code = np.zeros(self.metadata["n_attributes"])
        q_code[-1] = TARG_ENCODING

        return q_code
