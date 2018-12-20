import numpy as np


class ParallelComposition(object):

    def __init__(self):

        self.estimators_ = []

        self.desc_ids = np.array([])
        self.targ_ids = np.array([])

        self.classes_ = [np.array([])]

        self.n_classes_ = 0
        self.n_outputs_ = 0
        self.n_features_ = 0

        return

    def fit(self, X, Y, **kwargs):
        return

    def predict_proba(self, X, **kwargs):
        nb_rows, nb_atts = X.shape

        s_proba = [np.zeros(nb_rows, n) for n in self.n_classes_]

        for e in self.estimators_:
            e_proba = self._predict_proba_estimator_tidy(e, X, **kwargs)
            s_proba = self._add_proba_estimator(e, e_proba, s_proba)

        # redo sklearn convention from hell
        if len(s_proba) == 1:
            return s_proba[0]
        else:
            return s_proba

    def predict(self, X, **kwargs):
        nb_rows, nb_atts = X.shape

        s_pred = np.zeros(nb_rows, self.n_outputs_)

        # redo sklearn convention from hell
        if s_pred.shape[1] == 1:
            return s_pred.ravel()
        else:
            return s_pred

    # Updates (i.e., recalculate)
    def _update_classes_(self):
        # Re-initialize (easier)
        self.classes_ = [np.array([])] * len(self.targ_ids)

        for e in self.estimators_:
            self._add_classes_estimator(e)
        return

    def _update_n_classes_(self):
        self.n_classes_ = [len(c) for c in self.classes_]
        return

    def _update_n_outputs_(self):
        self.n_outputs_ = len(self.targ_ids)
        return

    def _update_n_features_(self):
        self.n_features = len(self.desc_ids)
        return

    # Add (i.e., incremental update)
    def add_estimator(self, e):
        self.estimators_.append(e)

        self._add_ids_estimator(e)

        self._update_classes_()

        self._update_n_classes_()
        self._update_n_outputs_()
        self._update_n_features_()

        return

    def _add_ids_estimator(self, e):
        self.desc_ids = np.unique(np.concatenate((self.desc_ids, e.desc_ids)))
        self.targ_ids = np.unique(np.concatenate((self.targ_ids, e.targ_ids)))
        return

    def _add_classes_estimator(self, e):

        idx_map = self._map_elements_idx(e.targ_ids, self.targ_ids)

        def combine(classes_1, classes_2):
            return np.unique(np.concatenate((classes_1, classes_2)))

        for idx_e, idx_s in idx_map:  # `s` stands for `self`
            e_classes_ = e.classes_[idx_e]
            s_classes_ = self.classes_[idx_s]

            self.classes_[idx_s] = combine(e_classes_, s_classes_)

        return

    def _add_proba_estimator(self, e, e_proba, s_proba):

        t_idx_map = self._map_elements_idx(e.targ_ids, self.targ_ids)

        for t_idx_e, t_idx_s in t_idx_map:  # `s` stands for `self`
            l_idx_map = self._map_elements_idx(e.classes_[t_idx_e], self.classes_[t_idx_s])
            l_idx_map = np.array(l_idx_map)

            l_idx_e, l_idx_s = l_idx_map[:, 0], l_idx_map[:, 1]

            s_proba[t_idx_s][:, l_idx_s] += e_proba[t_idx_e][:, l_idx_e]

        return s_proba

    # Utilities - Estimators
    @staticmethod
    def _predict_estimator_tidy(e, X, **kwargs):
        """
        Ensure matrix.
        """
        e_pred = e.predict(X, **kwargs)

        # undo sklearn convention from hell
        return np.atleast_2d(e_pred)

    @staticmethod
    def _predict_proba_estimator_tidy(e, X, **kwargs):
        """
        Ensure it is returned as a list.
        """
        e_proba = e.predict_proba(X, **kwargs)

        # undo sklearn convention from hell
        if isinstance(e_proba, np.ndarray):
            return [e_proba]
        elif isinstance(e_proba, list):
            return e_proba
        else:
            msg = """
            e_proba has to be (np.ndarray, list),
            instead the type was:   {}
            """.format(type(e_proba))
            raise TypeError(msg)

    # Utilities - Bookkeeping
    @staticmethod
    def _map_elements_idx(a1, a2):
        """
        Create a map that connects elements that occur in both arrays.

        The output is a tuple list, with a tuple being;
            (index of element e in a1, index of element e in a2)

        N.b.:   Does not crash in case of double entries (behaviour is still correct),
                but there are some ambiguities involved. I.e., do not do this.
        """
        idx_a1 = np.where(np.in1d(a1, a2))[0]
        idx_a2 = np.where(np.in1d(a2, a1))[0]

        return list(zip(idx_a1, idx_a2))
