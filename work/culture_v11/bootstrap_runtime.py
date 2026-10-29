"""Rebuild private runtime resources on a fresh server; no rating redistribution.

Run download.py first with the same V11_PRIVATE. This builds only runtime inputs,
never creates test labels or alters the frozen experiment implementation.
"""
import collections,json,random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from runtime_common import ROOT,PRIVATE,SEED,read,save
from identity import title_parts,match_entity
from tastetrail.qloo_http import QlooClient
from tastetrail.cluster_tools import build
from train_tools import extend

def main():
    destinations=['metadata.json','catalog.json','tools.npz']
    if any((PRIVATE/n).exists() for n in destinations):
        raise SystemExit('Runtime already exists; choose a new V11_PRIVATE instead of overwriting it.')
    ids=read(ROOT/'catalog_movie_ids.json')['movie_ids'];want=set(ids);meta={}
    for line in (PRIVATE/'movies.dat').read_text(encoding='latin1').splitlines():
        i,title,gs=line.split('::');i=int(i)
        if i in want:
            name,year=title_parts(title);meta[str(i)]={'title':name,'year':year,'genres':gs.split('|')}
    assert len(meta)==len(ids)
    def resolve(i):
        m=meta[str(i)];client=QlooClient(ttl=30*86400)
        e=match_entity(m,client.search(m['title']).get('results',[]))
        if not e:e=match_entity(m,client.search(f"{m['title']} {m['year']}").get('results',[]))
        return str(i),e['entity_id'] if e else None
    with ThreadPoolExecutor(max_workers=4) as ex:resolved=dict(ex.map(resolve,ids))
    missing=[i for i,q in resolved.items() if q is None]
    if missing:raise SystemExit(f'Cannot verify {len(missing)} movie identities; no incomplete runtime written.')
    assert len(set(resolved.values()))==len(ids)
    users=collections.defaultdict(list)
    for line in (PRIVATE/'ratings.dat').read_text().splitlines():
        u,i,r,t=map(int,line.split('::'));users[u].append((i,r))
    allusers=sorted(users);random.Random(20261024).shuffle(allusers);bg=sorted(set(users)-set(allusers[:2400]));idx={i:j for j,i in enumerate(ids)}
    raw=np.zeros((len(bg),len(ids)));means=np.zeros(len(bg))
    for j,u in enumerate(bg):
        means[j]=np.mean([r for i,r in users[u]])
        for i,r in users[u]:
            if i in idx:raw[j,idx[i]]=r
    genres=sorted({g for m in meta.values() for g in m['genres']})
    G=np.array([[int(g in meta[str(i)]['genres']) for g in genres] for i in ids],float)
    G/=np.maximum(np.linalg.norm(G,axis=1,keepdims=True),1e-9)
    save(PRIVATE/'metadata.json',meta);save(PRIVATE/'catalog.json',resolved)
    arrays=build(raw,means,G);arrays['ids']=np.array(ids);arrays=extend(arrays)
    np.savez_compressed(PRIVATE/'tools.npz',**arrays)
    (PRIVATE/'tools.npz').chmod(0o600)
    print(json.dumps({'ready':True,'movies':len(ids),'background_users':len(bg)}))

if __name__=='__main__':main()
