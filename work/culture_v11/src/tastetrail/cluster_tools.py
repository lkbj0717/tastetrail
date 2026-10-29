"""Preference reconstruction on observed ratings; shared offline/live code."""
import numpy as np
from .cohort_base import *
from .cohort_base import TasteTools as CohortTools
class TasteTools(CohortTools):
 def evaluate(self,history,candidates,qrows=None):
  values,qfeatures,trace=super().evaluate(history,candidates,qrows)
  a=self.a;hi=[self.idx[i] for i,r in history];ci=[self.idx[i] for i in candidates];ratings=np.array([r for i,r in history],float);res=ratings-ratings.mean()-a['item_bias'][hi]
  for k,reg in [(16,10),(32,10),(32,30)]:
   V=a[f'mf_{k}_{reg}'];H=V[hi];u=np.linalg.solve(H.T@H+reg*np.eye(k),H.T@res)
   scores=a['item_bias'][ci]+V[ci]@u;values[f'mf_{k}_{reg}']=float(scores[0]-scores[1])
  for reg in [10,100,1000]:
   scores=(ratings-3)@a[f'ease_{reg}'][np.ix_(hi,ci)];values[f'ease_{reg}']=float(scores[0]-scores[1])
  if qrows is not None:
   V=a['mf_32_10'];pred=[]
   for j in range(len(hi)):
    keep=np.arange(len(hi))!=j;h=np.array(hi)[keep];rr=ratings[keep];H=V[h];y=rr-rr.mean()-a['item_bias'][h]
    u=np.linalg.solve(H.T@H+10*np.eye(32),H.T@y);pred.append(rr.mean()+a['item_bias'][hi[j]]+V[hi[j]]@u)
   reliability=1/(1+(np.array(pred)-ratings)**2);signed=ratings-3;positive=ratings>=4;negative=ratings<=2
   logs=np.log(np.maximum(np.array(qrows)[1:],1e-9));ratio=logs[:,0]-logs[:,1]
   qfeatures['mf_weighted_qloo']=float(np.average(ratio[positive],weights=reliability[positive]*signed[positive])-np.average(ratio[negative],weights=reliability[negative]*(-signed[negative])))
  return values,qfeatures,trace
