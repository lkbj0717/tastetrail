import numpy as np
from affinity_graph import edge_loss_gradient,SparseAffinityTools

def test_observed_edge_gradient_only():
    rng=np.random.default_rng(123);params=[rng.normal(0,.2,(5,3)),rng.normal(0,.1,5),np.array([-.3])]
    ii=np.array([0,1,2]);jj=np.array([1,2,0]);y=np.array([-.2,-.4,-.3])
    loss,grad=edge_loss_gradient(params,ii,jj,y)
    assert np.all(grad[0][3:]==0) and np.all(grad[1][3:]==0)
    eps=1e-6
    for p,g in zip(params,grad):
        for index in list(np.ndindex(p.shape))[:10]:
            old=p[index];p[index]=old+eps;a=edge_loss_gradient(params,ii,jj,y)[0];p[index]=old-eps;b=edge_loss_gradient(params,ii,jj,y)[0];p[index]=old
            assert abs((a-b)/(2*eps)-g[index])<2e-7

def test_unknown_candidate_is_fallback_and_order_is_symmetric(tmp_path):
    rng=np.random.default_rng(3);E=rng.normal(size=(11,3));E/=np.linalg.norm(E,axis=1,keepdims=True);K=E@E.T;active=np.ones(11,bool);active[-1]=False
    p=tmp_path/'tools.npz';np.savez(p,movie_ids=np.arange(11),kernel=K,background_kernel=K,active=active,item_bias=np.zeros(11));tool=SparseAffinityTools(p)
    h=[(i,5 if i<5 else 1) for i in range(8)]
    a=tool.evaluate(h,[8,9]);b=tool.evaluate(h,[9,8]);assert all(abs(a[n]+b[n])<1e-12 for n in a)
    assert tool.evaluate(h,[8,10]) is None
