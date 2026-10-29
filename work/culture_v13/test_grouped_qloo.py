import numpy as np
from grouped_qloo import partition,new_features,features_from_rows

def test_partition_is_history_order_independent_and_candidate_free():
    K=np.eye(5);K[0,1]=K[1,0]=.9
    for i in [2,3,4]:
        for j in [2,3,4]:K[i,j]=.8 if i!=j else 1
    likes=[(i,4+i%2) for i in range(5)];idx={i:i for i in range(5)}
    a=partition(likes,K,idx);b=partition(list(reversed(likes)),K,idx)
    assert a==b;assert [i for i,r in a['groups'][0]]==[0,1]
    assert abs(sum(a['group_weights'])-1)<1e-12

def test_group_features_reverse_and_missing_is_not_zero():
    rows=[[.8,.7],[.6,.9],[.7,.5]];a=new_features(rows,[.4,.6],.2)
    b=new_features([r[::-1] for r in rows],[.4,.6],-.2)
    assert all(abs(a[k]+b[k])<1e-12 for k in a)
    try:new_features([[None,.7],[.6,.9],[.7,.5]],[.4,.6],.2)
    except ValueError:pass
    else:raise AssertionError('Missing affinity must reject evidence')

def test_public_feature_interface_aligns_candidate_order():
    history=[(i,5 if i<5 else 1) for i in range(8)]
    record={'error':None,'history':history,'candidates':[8,9],
      'plan':{'groups':[history[:2],history[2:5]],'coherence':[.9,.8]},
      'rows':[[.8,.7],[.6,.9],[.7,.5]]}
    old={'candidates':[8,9],'rows':[[.75,.65]]}
    a=features_from_rows(history,[8,9],record,old);b=features_from_rows(history,[9,8],record,old)
    assert all(abs(a[n]+b[n])<1e-12 for n in a)
