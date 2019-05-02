import numpy as np

from .Composition import Composition
from morpheus.utils import debug_print

VERBOSITY = 1


class SequentialComposition(Composition):
    def __init__(self):
        super().__init__()

        self.all_desc_ids = np.array([])
        return

    def predict(self, X, **kwargs):

        n_rows, n_atts = X.shape
        s_pred = np.zeros((n_rows, self.n_outputs_))
        D = np.empty((n_rows, len(self.all_desc_ids)))  # D is extended input matrix

        idx_map = self._map_elements_idx(
            self.desc_ids, self.all_desc_ids, return_array=True
        )
        X_idx, D_idx = idx_map[:, 0], idx_map[:, 1]
        D[:, D_idx] = X[:, X_idx]  # We fill up some entries of the D-matrix.

        for e in self.estimators_:
            idx_map = self._map_elements_idx(
                e.desc_ids, self.all_desc_ids, return_array=True
            )
            d_idx = idx_map[:, 1]
            e_outcome = self._predict_estimator_tidy(e, D[:, d_idx], **kwargs)

            msg = """
            e_outcome.shape:    {}
            """.format(
                e_outcome.shape
            )
            debug_print(msg, V=VERBOSITY)

            c_idx_map = self._map_elements_idx(
                e.targ_ids, self.all_desc_ids, return_array=True
            )  # Map of connections

            # If I predict one of the connections
            if c_idx_map.size > 0:
                c_idx_e, c_idx_s = c_idx_map[:, 0], c_idx_map[:, 1]
                D[:, c_idx_s] = e_outcome[:, c_idx_e]

            t_idx_map = self._map_elements_idx(
                e.targ_ids, self.targ_ids, return_array=True
            )  # Map of targets

            # If I predict one of the targets
            if t_idx_map.size > 0:
                msg = """
                t_idx_map:  {}
                """.format(
                    t_idx_map
                )
                debug_print(msg, V=VERBOSITY)

                t_idx_e, t_idx_s = t_idx_map[:, 0], t_idx_map[:, 1]
                s_pred[:, t_idx_s] = e_outcome[:, t_idx_e]

        if s_pred.shape[1] == 1:
            return s_pred.ravel()
        else:
            return s_pred

    def predict_numeric(self, X, **kwargs):

        n_rows, n_atts = X.shape
        s_numeric = np.zeros((n_rows, len(self.numeric_targ_ids)))
        s_weights = [
            t_weight
            for t_idx, t_weight in enumerate(self.targ_weights)
            if self.targ_types[t_idx] == "numeric"
        ]
        D = np.empty((n_rows, len(self.all_desc_ids)))  # D is extended input matrix

        idx_map = self._map_elements_idx(
            self.desc_ids, self.all_desc_ids, return_array=True
        )
        X_idx, D_idx = idx_map[:, 0], idx_map[:, 1]
        D[:, D_idx] = X[:, X_idx]  # We fill up some entries of the D-matrix.

        for e in self.estimators_:
            idx_map = self._map_elements_idx(
                e.desc_ids, self.all_desc_ids, return_array=True
            )
            d_idx = idx_map[:, 1]

            c_idx_map = self._map_elements_idx(
                e.targ_ids, self.all_desc_ids, return_array=True
            )  # Map of connections

            # If I predict one of the connections
            if c_idx_map.size > 0:
                e_outcome = self._predict_estimator_tidy(e, D[:, d_idx], **kwargs)
                c_idx_e, c_idx_s = c_idx_map[:, 0], c_idx_map[:, 1]
                D[:, c_idx_s] = e_outcome[:, c_idx_e]

            t_idx_map = self._map_elements_idx(
                e.targ_ids, self.numeric_targ_ids, return_array=True
            )  # Map of targets

            # If I predict one of the targets
            if t_idx_map.size > 0:
                e_numeric = self._predict_numeric_estimator_tidy(
                    e, D[:, d_idx], **kwargs
                )
                s_numeric = self._add_numeric_estimator_outcomes(
                    e, e_numeric, s_numeric
                )

        # Normalize
        s_numeric /= s_weights

        if s_numeric.shape[1] == 1:
            return s_numeric.ravel()
        else:
            return s_numeric

    def predict_nominal(self, X, **kwargs):

        n_rows, n_atts = X.shape
        s_nominal = [np.zeros((n_rows, n_clas)) for n_clas in self.n_classes_]
        s_weights = [
            t_weight
            for t_idx, t_weight in enumerate(self.targ_weights)
            if self.targ_types[t_idx] == "nominal"
        ]
        D = np.empty((n_rows, len(self.all_desc_ids)))  # D is extended input matrix

        idx_map = self._map_elements_idx(
            self.desc_ids, self.all_desc_ids, return_array=True
        )
        X_idx, D_idx = idx_map[:, 0], idx_map[:, 1]
        D[:, D_idx] = X[:, X_idx]  # We fill up some entries of the D-matrix.

        for e in self.estimators_:
            idx_map = self._map_elements_idx(
                e.desc_ids, self.all_desc_ids, return_array=True
            )
            d_idx = idx_map[:, 1]

            c_idx_map = self._map_elements_idx(
                e.targ_ids, self.all_desc_ids, return_array=True
            )  # Map of connections

            # If I predict one of the connections
            if c_idx_map.size > 0:
                e_outcome = self._predict_estimator_tidy(e, D[:, d_idx], **kwargs)
                c_idx_e, c_idx_s = c_idx_map[:, 0], c_idx_map[:, 1]
                D[:, c_idx_s] = e_outcome[:, c_idx_e]

            t_idx_map = self._map_elements_idx(
                e.targ_ids, self.nominal_targ_ids, return_array=True
            )  # Map of targets

            # If I predict one of the targets
            if t_idx_map.size > 0:
                e_nominal = self._predict_nominal_estimator_tidy(
                    e, D[:, d_idx], **kwargs
                )
                s_nominal = self._add_nominal_estimator_outcomes(
                    e, e_nominal, s_nominal
                )

        # Normalize
        s_nominal = [
            s_nominal[t_idx] / s_weights[t_idx]
            for t_idx in range(len(self.nominal_targ_ids))
        ]

        # redo sklearn convention from hell
        if len(s_nominal) == 1:
            return s_nominal[0]
        else:
            return s_nominal

    # Add (i.e., incremental update)
    def _add_estimator(self, e, location="out"):
        def check_connection(model_a, model_b):
            connecting_attributes = np.intersect1d(model_a.targ_ids, model_b.desc_ids)
            msg = """
            Connecting attributes:  {}
            """.format(
                connecting_attributes
            )
            debug_print(msg, V=VERBOSITY)
            return connecting_attributes.size > 0

        if len(self.estimators_) == 0:
            # No estimator yet, everything is OK.
            self.estimators_.insert(0, e)
        elif location in {"out", "output", "append", "back", "end"}:
            msg = """
            Trying to add a model to end of the chain.
            
            Current chain targ_ids:     {}
            New estimator desc_ids:     {}
            """.format(
                self.targ_ids, e.desc_ids
            )
            debug_print(msg, V=VERBOSITY)

            if check_connection(self, e):
                self.estimators_.append(e)
            else:
                msg = """
                Failed to connect the new estimator to the existing chain.
                
                Current chain has target attributes:        {}
                New estimator has descriptive attributes:   {}
                
                Since you decided to add this estimator to the end of the 
                current chain, there should be an overlap between the two
                in order to connect them. This is not the case.
                """.format(
                    self.targ_ids, e.desc_ids
                )
                raise ValueError(msg)
        elif location in {"in", "input", "prepend", "front", "begin"}:
            if check_connection(e, self):
                self.estimators_.insert(0, e)
            else:
                msg = """
                Failed to connect the new estimator to the existing chain.

                New estimator has target attributes:        {}
                Current chain has descriptive attributes:   {}
                
                Since you decided to add this estimator to the beginning of the 
                current chain, there should be an overlap between the two
                in order to connect them. This is not the case.
                """.format(
                    e.desc_ids, self.targ_ids
                )
                raise ValueError(msg)
        else:
            msg = """
            An estimator can only be added to a sequential composition if at 
            least one of its input attributes is an output attribute of the
            current sequential composition so far.
            
            Input attributes new estimator:                     {}
            Output attributes current sequential composition:   {}
            """.format(
                e.desc_ids, self.targ_ids
            )
            raise ValueError(msg)

        return

    def _add_ids_estimator(self, e):
        conn_ids = np.intersect1d(self.targ_ids, e.desc_ids)

        self.all_desc_ids = np.unique(np.concatenate((self.all_desc_ids, e.desc_ids)))
        self.desc_ids = np.unique(np.concatenate((self.desc_ids, e.desc_ids)))
        self.targ_ids = np.unique(np.concatenate((self.targ_ids, e.targ_ids)))

        # Remove the connection ids
        self.desc_ids = self.desc_ids[~np.in1d(self.desc_ids, conn_ids)]
        self.targ_ids = self.targ_ids[~np.in1d(self.targ_ids, conn_ids)]
        return
