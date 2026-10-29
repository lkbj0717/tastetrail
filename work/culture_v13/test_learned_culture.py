"""Meaningful derivative, symmetry, and explicit-feedback boundary tests."""
import numpy as np
from learned_culture import prepare_context,forward,regularization,_draw

def fixture():
    rng=np.random.default_rng(44);Z=rng.normal(size=(12,4));Z/=np.linalg.norm(Z,axis=1,keepdims=True)
    bias=rng.normal(size=12);h=np.array([[0,1,2,3,4,5,6,7]]);r=np.array([[5,4,4,5,4,1,2,1]])
    c=np.array([[8,9]]);params=[np.eye(4)+rng.normal(0,.1,(4,4)),np.eye(4),rng.normal(0,.1,4),np.ones(1)]
    return Z,bias,h,r,c,params

def test_finite_difference_gradients():
    Z,bias,h,r,c,params=fixture();ctx=prepare_context(Z,bias,h,r,c);_,grads=forward(params,ctx,True)
    def loss():return float(np.logaddexp(0,-forward(params,ctx)).mean())
    eps=1e-6
    for p,g in zip(params,grads):
        for index in list(np.ndindex(p.shape))[:8]:
            original=p[index];p[index]=original+eps;a=loss();p[index]=original-eps;b=loss();p[index]=original
            assert abs((a-b)/(2*eps)-g[index])<2e-7

def test_candidate_swap_and_history_permutation():
    Z,bias,h,r,c,params=fixture();a=forward(params,prepare_context(Z,bias,h,r,c))
    b=forward(params,prepare_context(Z,bias,h,r,c[:,::-1]));assert np.allclose(a,-b,atol=1e-12)
    perm=[4,0,2,1,3,7,5,6];v=forward(params,prepare_context(Z,bias,h[:,perm],r[:,perm],c))
    assert np.allclose(a,v,atol=1e-12)

def test_sampler_uses_only_explicit_temporal_pools():
    p={'history_pos':np.arange(5),'history_neg':np.arange(5,8),'target_pos':np.array([8]),'target_neg':np.array([9]),'ratings':{**{i:5 for i in range(5)},**{i:1 for i in range(5,8)},8:4,9:2}}
    h,r,c=_draw([p],20,np.random.default_rng(1),np.ones((12,12)),np.zeros(12))
    assert set(h.ravel())==set(range(8));assert set(c.ravel())=={8,9}
    assert np.all(r[:,:5]>=4) and np.all(r[:,5:]<=2)
    assert 10 not in h and 10 not in c  # Missing rating is never negative.
