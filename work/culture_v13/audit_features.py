"""Public Qloo entity metadata -> label-free cultural preference tools.

No user IDs, hidden candidate ratings, or formal-set outcomes enter this module.
Raw metadata is stored outside the public source tree. No network calls needed.
"""
from pathlib import Path
import collections, json, re, math, time
import numpy as np

PRIVATE = Path.home()/'.local/share/tastetrail/culture_v13'
OLD = PRIVATE.parent/'culture_v11'
FIELDS = ['genres','keywords','collaborators','duration','content_rating',
          'release_country','plot_summary','audience_identity','genre_description',
          'style_description','plot_themes_description','emotional_tone_description',
          'situational_contexts']
TEXT_FIELDS = ['audience_identity','style_description','plot_themes_description',
               'emotional_tone_description','genre_description']

def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)); path.chmod(0o600)

def extract():
    """Use cache responses as movie metadata only, irrespective of query origin."""
    catalog=json.loads((OLD/'catalog.json').read_text())
    reverse={qid:mid for mid,qid in catalog.items()}; found={}; quality={}
    cache=PRIVATE.parent/'qloo_cache'
    for path in sorted(cache.glob('*.json')):
        try:
            obj=json.loads(path.read_text()); data=obj.get('data',{}); results=data.get('results',{})
            entities=results if isinstance(results,list) else results.get('entities',[])
            for e in entities:
                mid=reverse.get(e.get('entity_id'))
                if mid is None: continue
                p=e.get('properties') or {}; tags=e.get('tags') or []
                score=(sum(bool(p.get(f)) for f in FIELDS),len(tags),obj.get('fetched_at',0))
                if score<=quality.get(mid,(-1,-1,-1)):continue
                quality[mid]=score
                found[mid]={'entity_id':e['entity_id'],'name':e.get('name'),
                    'release_year':p.get('release_year'),
                    'properties':{f:p[f] for f in FIELDS if p.get(f) is not None},
                    'tags':[{'id':t.get('id') or t.get('tag_id'),
                             **{k:t[k] for k in ['name','type'] if k in t}} for t in tags],
                    'source_cache_digest':path.stem,'fetched_at':obj.get('fetched_at')}
        except (ValueError,KeyError,TypeError):continue
    missing=sorted(set(catalog)-set(found),key=int)
    out={'schema_version':1,'source':'Qloo cached public movie entity metadata',
         'excluded':['affinity','hidden_ratings','user IDs','external aggregate ratings'],
         'movie_count':len(found),'missing_movie_ids':missing,'movies':found}
    save(PRIVATE/'qloo_metadata.json',out)
    summary={'movies':len(found),'missing':missing,'properties_coverage':{
        f:sum(bool(e['properties'].get(f)) for e in found.values()) for f in FIELDS},
        'tag_types':dict(collections.Counter(t.get('type') for e in found.values() for t in e['tags']))}
    save(Path(__file__).parent/'research/metadata_coverage.json',summary)
    return summary

STOP=set('a an and are as at be by for from has have in into is it its of on or that the their these this through to with who which often people viewers film films movie movies story stories'.split())

def _tokens(e,kind):
    p=e['properties']
    if kind in ['tags','curated']:
        return {t['id'] for t in e['tags'] if t.get('id') and (kind=='tags' or ':qloo:' in t['id'])}
    if kind=='text':
        text=' '.join(str(p.get(f,'')) for f in TEXT_FIELDS)
        return {w for w in re.findall(r'[a-z][a-z]+',text.lower()) if w not in STOP}
    if kind=='people':return {x.casefold() for x in p.get('collaborators',[])}
    raise ValueError(kind)

def _kernel(entries,kind):
    tokens=[_tokens(e,kind) for e in entries]
    counts=collections.Counter(t for row in tokens for t in row)
    vocab=sorted(t for t,n in counts.items() if n>= (1 if kind=='people' else 2) and n<=len(tokens)*.95)
    lookup={t:i for i,t in enumerate(vocab)}
    X=np.zeros((len(entries),len(vocab)),dtype=float)
    for i,row in enumerate(tokens):
        for t in row:
            if t in lookup:X[i,lookup[t]]=math.log((1+len(entries))/(1+counts[t]))+1
    X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    return X@X.T, len(vocab)

class CulturalMetadataTools:
    """TF-IDF culture-tag and text kernels; 8-rating ridge fold-in per user."""
    def __init__(self,path=None):
        obj=json.loads((Path(path) if path else PRIVATE/'qloo_metadata.json').read_text())
        self.ids=sorted(map(int,obj['movies']));self.index={i:j for j,i in enumerate(self.ids)}
        entries=[obj['movies'][str(i)] for i in self.ids];self.kernels={};self.vocab_sizes={}
        for kind in ['tags','curated','text','people']:
            self.kernels[kind],self.vocab_sizes[kind]=_kernel(entries,kind)

    def evaluate(self,history,candidates):
        hi=[self.index[int(i)] for i,r in history];ci=[self.index[int(i)] for i in candidates]
        r=np.array([r for i,r in history],float);pos=r>=4;neg=r<=2
        weights=np.abs(r-3);y=r-r.mean();out={}
        for kind,K in self.kernels.items():
            sim=K[np.ix_(hi,ci)]
            scores=np.average(sim[pos],axis=0,weights=weights[pos])-np.average(sim[neg],axis=0,weights=weights[neg])
            out[f'qmeta_{kind}_contrast']=float(scores[0]-scores[1])
            gram=K[np.ix_(hi,hi)]
            for ridge in [.1,1.,10.]:
                prediction=sim.T@np.linalg.solve(gram+ridge*np.eye(len(hi)),y)
                out[f'qmeta_{kind}_ridge_{ridge:g}']=float(prediction[0]-prediction[1])
        return out

if __name__=='__main__':
    print(json.dumps(extract(),ensure_ascii=False))
