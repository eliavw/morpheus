import numpy as np

from .Composition import Composition
from ..utils.extra import debug_print

VERBOSITY = 1


class SequentialComposition(Composition):

    def __init__(self):
        super().__init__()

        self.all_desc_ids = np.array([])
        return

    def predict(self, X, **kwargs):

        n_rows, n_atts = X.shape
        Y = np.zeros((n_rows, len(self.targ_ids)))
        D = np.empty((n_rows, len(self.all_desc_ids))) # D is extended input matrix

        idx_map = self._map_elements_idx(self.desc_ids, self.all_desc_ids, return_array=True)
        X_idx, D_idx = idx_map[:, 0], idx_map[:, 1]
        D[:, D_idx] = X[:, X_idx] # We fill up some entries of the D-matrix.

        for e in self.estimators_:
            idx_map = self._map_elements_idx(e.desc_ids, self.all_desc_ids, return_array=True)
            d_idx = idx_map[:, 1]
            e_outcome = self._predict_estimator_tidy(e, D[:, d_idx], **kwargs)

            msg = """
            e_outcome.shape:    {}
            """.format(e_outcome.shape)
            debug_print(msg, V=VERBOSITY)

            c_idx_map = self._map_elements_idx(e.targ_ids, self.all_desc_ids, return_array=True)    # Map of connections
            if c_idx_map.size > 0:
                c_idx_e, c_idx_s = c_idx_map[:, 0], c_idx_map[:, 1]
                D[:, c_idx_s] = e_outcome[:, c_idx_e]

            t_idx_map = self._map_elements_idx(e.targ_ids, self.targ_ids, return_array=True)        # Map of targets
            if t_idx_map.size > 0:
                msg = """
                t_idx_map:  {}
                """.format(t_idx_map)
                debug_print(msg, V=VERBOSITY)

                t_idx_e, t_idx_s = t_idx_map[:, 0], t_idx_map[:, 1]
                Y[:, t_idx_s] = e_outcome[:, t_idx_e]

        if Y.shape[1] == 1:
            return Y.ravel()
        else:
            return Y

    # Add (i.e., incremental update)
    def _add_estimator(self, e):

        if len(self.estimators_) == 0:
            # No estimator yet, everything is OK.
            self.estimators_.append(e)
        elif np.intersect1d(self.targ_ids, e.desc_ids).size != 0:
            # New estimator connects to outputs of previous estimator
            self.estimators_.append(e)
        else:
            msg = """
            An estimator can only be added to a sequential composition if at 
            least one of its input attributes is an output attribute of the
            current sequential composition so far.
            
            Input attributes new estimator:                     {}
            Output attributes current sequential composition:   {}
            """.format(e.desc_ids, self.targ_ids)
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

