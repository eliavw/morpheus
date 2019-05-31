import numpy as np

from morpheus.algo.inference import get_predict
from morpheus.graph import get_ids


class CompositeModel(object):
    def __init__(self, diagram, methods, targ_ids=None):
        self.desc_ids = list(get_ids(diagram, kind='desc'))

        if targ_ids is None:
            self.targ_ids = list(get_ids(diagram, kind='targ'))
        else:
            self.targ_ids = targ_ids

        self.feature_importances_ = self.extract_feature_importances(diagram)

        self.predict = get_predict(methods, self.targ_ids)

        return

    def extract_feature_importances(self, diagram, aggregation=np.sum):
        fi = []
        for idx in self.desc_ids:
            fi_idx = [d.get('fi', 0)
                      for src, tgt, d in diagram.edges(data=True)
                      if d.get('idx', 0) == idx]
            fi.append(aggregation(fi_idx))

        norm = np.linalg.norm(fi, 1)
        fi = fi / norm
        return fi
