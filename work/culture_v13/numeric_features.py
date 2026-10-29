"""Reusable local tools; explicit observations only, no access to case labels."""
import numpy as np
from common import OLD, recovered_output
from tastetrail.cluster_tools import TasteTools
from tastetrail.fusion import qloo_features
from audit_features import CulturalMetadataTools
from graph_models import GraphTools

QBASE=['pooled_log_ratio','signed_history_log_ratio','best_like_log_ratio','signed_agreement']


class FeatureBuilder:
    def __init__(self):
        self.community=TasteTools(dict(np.load(OLD/'tools.npz')))
        self.culture=CulturalMetadataTools()
        self.graph=GraphTools()

    def build(self,history,candidates,initial,evidence=None):
        output=recovered_output(initial)
        rows=evidence.get('rows') if evidence and not evidence.get('error') else None
        values,extra,trace=self.community.evaluate(history,candidates,rows)
        values['glm_log_odds']=output['log_odds'] if output else 0.
        values['cohort_gated_glm']=values['glm_log_odds']*(1-values['cohort_entropy'])
        if rows is not None:
            values.update(dict(zip(QBASE,qloo_features(rows,[r for i,r in history]))))
            values.update(extra)
        else:
            values.update({k:0. for k in QBASE+['mf_weighted_qloo','cohort_weighted_qloo']})
        values.update(self.culture.evaluate(history,candidates))
        graph,gtrace=self.graph.evaluate(history,candidates)
        values.update(graph)
        return {'values':values,'trace':{'community':trace,'culture_graph':gtrace},
                'initial_valid':output is not None,'affinity_valid':rows is not None}
