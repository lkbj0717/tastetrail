import numpy as np
import pytest
from tastetrail.cluster_tools import build,TasteTools,interest_groups

@pytest.fixture(scope='module')
def tools():
 rng=np.random.default_rng(123);raw=rng.integers(1,6,size=(64,12)).astype(float);raw[rng.random(raw.shape)<.2]=0
 means=raw.sum(axis=1)/(raw>0).sum(axis=1);G=np.eye(4)[np.arange(12)%4]
 a=build(raw,means,G);a['ids']=np.arange(100,112)
 
 for k,reg in [(16,10),(32,10),(32,30)]:a[f'mf_{k}_{reg}']=rng.normal(size=(12,k))
 for reg in [10,100,1000]:a[f'ease_{reg}']=rng.normal(size=(12,12))
 return TasteTools(a)

def test_swap_candidates_reverses_preferences_not_membership(tools):
 h=[(100+i,r) for i,r in enumerate([5,4,5,4,4,1,2,1])];q=np.random.default_rng(9).uniform(.1,1,(9,2))
 v,qq,t=tools.evaluate(h,[108,109],q);vs,qs,ts=tools.evaluate(h,[109,108],q[:,::-1])
 for key in v:
  np.testing.assert_allclose(v[key],vs[key] if key=='cohort_entropy' else -vs[key],atol=1e-10)
 for key in qq:np.testing.assert_allclose(qq[key],-qs[key],atol=1e-10)
 assert t['interest_groups']==ts['interest_groups'] and t['top_cohorts']==ts['top_cohorts']

def test_cohort_weights_and_evidence_are_finite(tools):
 v,q,t=tools.evaluate([(100+i,r) for i,r in enumerate([5,4,5,4,4,1,2,1])],[108,109])
 assert all(np.isfinite(x) for x in v.values()) and q=={}
 assert all(x['background_members']>=0 for x in t['top_cohorts'])
 assert 0<=t['cohort_entropy']<=1+1e-12

def test_missing_or_invalid_qloo_is_not_zero_evidence(tools):
 with pytest.raises(ValueError):tools.evaluate([(100+i,r) for i,r in enumerate([5,4,5,4,4,1,2,1])],[108,109],[[.1,.2]])
