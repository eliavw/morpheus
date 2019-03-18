import numpy as np

from ..utils.extra import debug_print

VERBOSITY = 0


class Composition(object):
    def __init__(self):

        self.estimators_ = []

        self.desc_ids = np.array([])
        self.targ_ids = np.array([])

        self.classes_ = [np.array([])]

        self.n_classes_ = [-1]
        self.n_outputs_ = 0
        self.n_features_ = 0

        self.targ_types = []
        self.targ_weights = np.array([])

        return

    # Basics
    def fit(self, X, Y, **kwargs):
        return

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

    def predict_proba(self, X, **kwargs):
        return self.predict_nominal(X, **kwargs)

    def add_estimator(self, e):
        self._add_estimator(e)
        self._add_ids_estimator(e)

        self._update_targ_types()
        self._update_nominal_numeric_targ_ids()

        self._update_classes_()
        self._update_targ_weights()

        self._update_n_classes_()
        self._update_n_outputs_()
        self._update_n_features_()

        return

    def pretty_print(self):

        estimators_used = [e.__class__.__name__ for e in self.estimators_]

        estimator_summary = """
        ## Estimators
        Estimators used:        {}
        N_estimators:           {}
        """.format(estimators_used,
                   len(estimators_used))

        attr_summary = """
        # Main
        Descriptive attributes:     {}
        Target attributes:          {}

        ## On types (mostly nominals)
        Target attribute types:                     {}
        N_classes of nominal target attributes:     {}
        Classes of nominal target attributes:       {}

        ## Weights
        Total weights of target attributes:         {}

        """.format(self.desc_ids,
                   self.targ_ids,
                   self.targ_types,
                   self.n_classes_,
                   self.classes_,
                   self.targ_weights)

        print(attr_summary, estimator_summary)
        return

    # Updates
    def _update_targ_types(self):
        # Re-initialize (easier)
        self.targ_types = [None] * len(self.targ_ids)

        for e in self.estimators_:
            self._add_targ_types_estimator(e)

        return

    def _update_nominal_numeric_targ_ids(self):
        self.nominal_targ_ids = [t for t_idx, t in enumerate(self.targ_ids)
                                 if self.targ_types[t_idx] == 'nominal']
        self.numeric_targ_ids = [t for t_idx, t in enumerate(self.targ_ids)
                                 if self.targ_types[t_idx] == 'numeric']
        return

    def _update_classes_(self):
        # Re-initialize (easier)
        self.classes_ = [np.array([])] * len(self.nominal_targ_ids)

        for e in self.estimators_:
            self._add_classes_estimator(e)
        return

    def _update_targ_weights(self):
        """
        Set the weights for each target attribute.

        The weight counts how many estimators that predict this target,
        and as a consequence, with which factor the outcome should be
        normalized.

        Returns:

        """

        # Re-initialize (easier)
        self.targ_weights = np.zeros((len(self.targ_ids, )))

        for e in self.estimators_:
            self._add_targ_weights_estimator(e)
        return

    def _update_n_classes_(self):
        """
        Count the classes.

        This means that you count the amount of classes of each
        nominal output attribute.

        Returns:

        """
        self.n_classes_ = [len(c) for c in self.classes_]
        return

    def _update_n_outputs_(self):
        """
        Count the number of outputs.

        Returns:

        """
        self.n_outputs_ = len(self.targ_ids)
        return

    def _update_n_features_(self):
        """
        Count the number of input attributes.

        Returns:

        """
        self.n_features_ = len(self.desc_ids)
        return

    # Add (i.e., incremental update)
    def _add_targ_types_estimator(self, e):

        idx_map = self._map_elements_idx(e.targ_ids, self.targ_ids)

        for t_idx_e, t_idx_s in idx_map:
            self.targ_types[t_idx_s] = self._targ_types_estimator_tidy(e)[t_idx_e]

        return

    def _add_classes_estimator(self, e):

        idx_map = self._map_elements_idx(e.targ_ids, self.nominal_targ_ids)

        def combine(classes_1, classes_2):
            return np.unique(np.concatenate((classes_1, classes_2)))

        for idx_e, idx_s in idx_map:  # `s` stands for `self`
            e_classes_ = self._classes_estimator_tidy(e)[idx_e]
            s_classes_ = self.classes_[idx_s]

            msg = """
            e_classes:          {}
            s_classes:          {}

            type(e), type(s):   {}, {}
            """.format(e_classes_, s_classes_, type(e_classes_), type(s_classes_))
            debug_print(msg, V=VERBOSITY)

            self.classes_[idx_s] = combine(e_classes_, s_classes_)

        return

    def _add_targ_weights_estimator(self, e, weights=None):
        t_idx_map = self._map_elements_idx(e.targ_ids, self.targ_ids)

        if weights is None:
            # Assume defaults
            weights = [1] * len(e.targ_ids)
        elif isinstance(weights, (int, float)):
            # Assume uniform weights
            weights = [weights] * len(e.targ_ids)
        elif isinstance(weights, list):
            assert len(weights) == len(e.targ_ids)
        else:
            msg = """
            Weights has to be of type (None, int, float or list)
            Instead, we got: {}, {}
            """.format(weights, type(weights))
            ValueError(msg)

        for t_idx_e, t_idx_s in t_idx_map:  # `s` stands for `self`
            self.targ_weights[t_idx_s] += weights[t_idx_e]

        return

    def _add_nominal_estimator_outcomes(self, e, e_nominal, s_nominal):

        t_idx_map = self._map_elements_idx(e.targ_ids, self.nominal_targ_ids)

        for t_idx_e, t_idx_s in t_idx_map:  # `s` stands for `self`

            t_classes_e = self._classes_estimator_tidy(e)[t_idx_e]
            t_classes_s = self.classes_[t_idx_s]

            l_idx_map = self._map_elements_idx(t_classes_e, t_classes_s, return_array=True) # Labels map

            msg = """
            l_idx_map:  {}
            """.format(l_idx_map)
            debug_print(msg, V=VERBOSITY)

            l_idx_e, l_idx_s = l_idx_map[:, 0], l_idx_map[:, 1]

            msg = """
            t_idx_e, t_idx_s: {}, {}
            l_idx_e, l_idx_s: {}, {}
            """.format(t_idx_e, t_idx_s, l_idx_e, l_idx_s)
            debug_print(msg, V=VERBOSITY)

            s_nominal[t_idx_s][:, l_idx_s] += e_nominal[t_idx_e][:, l_idx_e]

        return s_nominal

    def _add_numeric_estimator_outcomes(self, e, e_numeric, s_numeric):

        t_idx_map = self._map_elements_idx(e.targ_ids, self.numeric_targ_ids, return_array=True)
        t_idx_e, t_idx_s = t_idx_map[:, 0], t_idx_map[:, 1]

        msg = """
        t_idx_e, t_idx_s:    {}, {}
        """.format(t_idx_e, t_idx_s)
        debug_print(msg, V=VERBOSITY)

        s_numeric[:, t_idx_s] += e_numeric[:, t_idx_e]

        return s_numeric

    def _add_outcome_estimator(self, e, e_outcome, s_outcome):

        t_idx_map = self._map_elements_idx(e.targ_ids, self.targ_ids, return_array=True)
        t_idx_e, t_idx_s = t_idx_map[:, 0], t_idx_map[:, 1]

        msg = """
        t_idx_e, t_idx_s:    {}, {}
        """.format(t_idx_e, t_idx_s)
        debug_print(msg, V=VERBOSITY)

        s_outcome[:, t_idx_s] += e_outcome[:, t_idx_e]

        return s_outcome

    # Utilities - Estimators
    @staticmethod
    def _predict_estimator_tidy(e, X, **kwargs):
        """
        Ensure matrix.
        """
        e_pred = e.predict(X, **kwargs)

        # undo sklearn convention from hell
        if e_pred.ndim < 2:
            return np.atleast_2d(e_pred).T
        else:
            return e_pred

    @staticmethod
    def _predict_nominal_estimator_tidy(e, X, **kwargs):
        """
        Ensure result as list.

        Sci-kit learn has the odd tendency to treat
        models that predict a single attribute different
        from multi-output models.

        Single outputs are returned as a numpy vector,
        whereas the in the case of multiple outputs, a list is returned.
        Due to dimensional issues, a list is the logical option. However,
        for consistency, we opt to also follow this when a single target is
        predicted.
        """

        e_nominal = e.predict_proba(X, **kwargs)

        # undo sklearn convention from hell
        if isinstance(e_nominal, np.ndarray):
            return [e_nominal]
        elif isinstance(e_nominal, list):
            return e_nominal
        else:
            msg = """
            e_nominal has to be (np.ndarray, list),
            instead the type was:   {}
            """.format(type(e_nominal))
            raise TypeError(msg)

    @staticmethod
    def _predict_numeric_estimator_tidy(e, X, **kwargs):
        """
        Ensure result as 2D numpy array.

        Sci-kit learn has the odd tendency to treat
        models that predict a single attribute different
        from multi-output models.

        Single outputs are returned as a numpy vector,
        whereas the in the case of multiple outputs, a 2D array is returned.
        """

        if hasattr(e, "predict_numeric"):
            # Own model => No problem
            e_numeric = e.predict_numeric(X, **kwargs)
        elif not hasattr(e, "classes_"):
            # Sklearn regressor => Also alright
            e_numeric = e.predict(X, **kwargs)
        else:
            msg = """
            No idea what kind of regressor you passed but computer says no.
            Either pure regressor (sklearn-like, no classes_ attribute), or 
            something that obeys MERCS' conventions and therefore has a
            predict_numeric method.

            So, either it is pure, or there is clarity;
            otherwise we cannot handle mixtures recursively.
            """
            raise TypeError(msg)

        # undo sklearn convention from hell
        if e_numeric.ndim < 2:
            return np.atleast_2d(e_numeric).T
        else:
            return e_numeric

    @staticmethod
    def _classes_estimator_tidy(e):

        e_classes_ = e.classes_

        # undo sklearn convention from hell
        if isinstance(e_classes_, np.ndarray):
            return [e_classes_]
        elif isinstance(e_classes_, list):
            return e_classes_
        else:
            msg = """
            e_classes_ has to be (np.ndarray, list),
            instead the type was:   {}
            """.format(type(e_classes_))
            raise TypeError(msg)

    @staticmethod
    def _targ_types_estimator_tidy(e):

        if hasattr(e, "targ_types"):
            # Own object => Possibly mixture, but we can just copy.
            targ_types = e.targ_types
        elif hasattr(e, "classes_"):
            # Sklearn + Classes => All nominal
            targ_types = ['nominal'] * e.n_outputs_
        elif not hasattr(e, "classes_"):
            # Sklearn + ~Classes => All numeric
            targ_types = ['numeric'] * e.n_outputs_
        else:
            msg = """
            Cannot handle what was passed. This method either takes
            a pure sklearn or a MERCS estimator.
            """.format(e)
            TypeError(e)

        return targ_types

    # Utilities - Bookkeeping
    @staticmethod
    def _map_elements_idx(a1, a2, return_array=False):
        """
        Create a map that connects elements that occur in both arrays.

        The output is a tuple list, with a tuple being;
            (index of element e in a1, index of element e in a2)

        N.b.:   Does not crash in case of double entries (behaviour is still correct),
                but there are some ambiguities involved. So: do not do this.
        """
        idx_a1 = np.where(np.in1d(a1, a2))[0]
        idx_a2 = np.where(np.in1d(a2, a1))[0]

        result = list(zip(idx_a1, idx_a2))

        if return_array:
            return np.atleast_2d(np.array(result))
        else:
            return result
