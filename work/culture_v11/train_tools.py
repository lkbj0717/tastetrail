"""Observed-only ALS, closed-form EASE, and factor-space taste cohorts."""
import time
import numpy as np
from tastetrail.cohort_base import spherical_kmeans

def extend(a):
 mask=a['mask'].astype(bool);Y=a['residual'];rows=[np.flatnonzero(x) for x in mask];cols=[np.flatnonzero(x) for x in mask.T]
 for k,reg in [(16,10),(32,10),(32,30)]:
  start=time.monotonic();V=a['item_factors'][:,:k].copy();I=np.eye(k)*reg;U=np.zeros((Y.shape[0],k))
  for step in range(20):
   for u,items in enumerate(rows):
    H=V[items];U[u]=np.linalg.solve(H.T@H+I,H.T@Y[u,items])
   for i,people in enumerate(cols):
    H=U[people];V[i]=np.linalg.solve(H.T@H+I,H.T@Y[people,i])
  a[f'mf_{k}_{reg}']=V
  print({'trained':f'mf_{k}_{reg}','seconds':round(time.monotonic()-start,2)},flush=True)
  if (k,reg)==(32,10):
   centers,labels=spherical_kmeans(U,32,seed=20261026);prototypes=[];sizes=[];centered=(a['raw']-a['user_means'][:,None])*mask
   for j in range(32):
    group=labels==j;sizes.append(int(group.sum()));prototypes.append((centered[group].sum(axis=0)+10*a['item_bias'])/(mask[group].sum(axis=0)+10))
   a.update(cohort_centers=centers,cohort_labels=labels,cohort_prototypes=np.array(prototypes),cohort_sizes=np.array(sizes))
 # Explicit signed preferences: no labels from prospective people.
 S=(a['raw']-3)*mask;G=S.T@S
 for reg in [10,100,1000]:
  P=np.linalg.inv(G+reg*np.eye(G.shape[0]));B=-P/np.diag(P)[None,:];np.fill_diagonal(B,0);a[f'ease_{reg}']=B
 return a
