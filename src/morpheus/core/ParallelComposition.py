import numpy as np

from .Composition import Composition
from ..utils.extra import debug_print

VERBOSITY = 0


class ParallelComposition(Composition):

    def __init__(self):
        super().__init__()
        return

    # Basics
    def predict(self, X, **kwargs):
        nb_rows, nb_atts = X.shape

        s_pred = np.zeros((nb_rows, self.n_outputs_))

        # Fill in numeric
        t_idx_map_numeric = self._map_elements_idx(self.numeric_targ_ids, self.targ_ids, return_array=True)
        t_idx_numeric, t_idx_s_numeric = t_idx_map_numeric[:, 0], t_idx_map_numeric[:, 1]

        s_numeric = self._predict_numeric_estimator_tidy(self, X, **kwargs)

        s_pred[:, t_idx_s_numeric] = s_numeric[:, t_idx_numeric]

        # Fill in nominal
        t_idx_map_nominal = self._map_elements_idx(self.nominal_targ_ids, self.targ_ids)

        s_nominal_proba = self._predict_nominal_estimator_tidy(self, X, **kwargs)

        for t_idx_nominal, t_idx_s_nominal in t_idx_map_nominal:
            s_pred[:, t_idx_s_nominal] = self.classes_[t_idx_nominal].take(np.argmax(s_nominal_proba[t_idx_nominal],
                                                                                     axis=1),
                                                                           axis=0)

        # redo sklearn convention from hell
        if s_pred.shape[1] == 1:
            return s_pred.ravel()
        else:
            return s_pred

    def predict_nominal(self, X, **kwargs):
        n_rows, n_atts = X.shape

        s_nominal = [np.zeros((n_rows, n_clas)) for n_clas in self.n_classes_]
        s_weights = [t_weight for t_idx, t_weight in enumerate(self.targ_weights)
                     if self.targ_types[t_idx] == "nominal"]

        def is_relevant(e):
            return np.intersect1d(e.targ_ids, self.nominal_targ_ids).shape[0] > 0

        relevant_estimators = [e for e in self.estimators_ if is_relevant(e)]

        for e in relevant_estimators:
            # Filter desc. atts
            d_idx_map = self._map_elements_idx(e.desc_ids, self.desc_ids, return_array=True)
            d_idx_comp = d_idx_map[:, 1]

            # Do prediction
            e_nominal = self._predict_nominal_estimator_tidy(e, X[:, d_idx_comp], **kwargs)
            s_nominal = self._add_nominal_estimator_outcomes(e, e_nominal, s_nominal)

        # Normalize
        s_nominal = [s_nominal[t_idx] / s_weights[t_idx]
                     for t_idx in range(len(self.nominal_targ_ids))]

        # redo sklearn convention from hell
        if len(s_nominal) == 1:
            return s_nominal[0]
        else:
            return s_nominal

    def predict_numeric(self, X, **kwargs):
        n_rows, n_atts = X.shape

        s_numeric = np.zeros((n_rows, len(self.numeric_targ_ids)))
        s_weights = [t_weight for t_idx, t_weight in enumerate(self.targ_weights)
                     if self.targ_types[t_idx] == "numeric"]

        def is_relevant(e):
            return np.intersect1d(e.targ_ids, self.numeric_targ_ids).shape[0] > 0

        relevant_estimators = [e for e in self.estimators_ if is_relevant(e)]

        for e in relevant_estimators:
            # Filter desc. atts
            d_idx_map = self._map_elements_idx(e.desc_ids, self.desc_ids, return_array=True)
            d_idx_comp = d_idx_map[:, 1]

            # Do prediction
            e_numeric = self._predict_numeric_estimator_tidy(e, X[:, d_idx_comp], **kwargs)
            s_numeric = self._add_numeric_estimator_outcomes(e, e_numeric, s_numeric)

        msg = """
        s_numeric.shape:    {}
        """.format(s_numeric.shape)
        debug_print(msg, V=VERBOSITY)

        # Normalize
        s_numeric /= s_weights

        # redo sklearn convention from hell
        if s_numeric.shape[1] == 1:
            return s_numeric.ravel()
        else:
            return s_numeric

    # Add (i.e., incremental update)
    def _add_estimator(self, e):
        self.estimators_.append(e)
        return

    def _add_ids_estimator(self, e):
        self.desc_ids = np.unique(np.concatenate((self.desc_ids, e.desc_ids)))
        self.targ_ids = np.unique(np.concatenate((self.targ_ids, e.targ_ids)))
        return
