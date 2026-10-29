import re,unicodedata
def title_parts(title):
 match=re.search(r'\((\d{4})\)\s*$',title)
 year=int(match.group(1)) if match else None
 title=title[:match.start()].strip() if match else title
 m=re.search(r', (The|A|An)$',title)
 if m:title=m[1]+' '+title[:m.start()]
 return title,year


def norm(s):return ''.join(c for c in unicodedata.normalize('NFKD',s).casefold() if c.isalnum())


def titles(s):
 out=[s,re.sub(r'\s*\([^)]*\)','',s).strip()]
 out+=re.findall(r'\(([^)]*)\)',s)
 return {norm(x) for x in out if x}


def match_entity(movie,entities):
 good=[]
 for e in entities:
  p=e.get('properties') or {}
  if 'urn:entity:movie' not in e.get('types',[]) or p.get('release_year')!=movie['year']:continue
  aliases=[e['name']]+[a['value'] for a in p.get('akas',[]) if isinstance(a,dict) and a.get('value')]
  if any(titles(movie['title']) & titles(a) for a in aliases):good.append(e)
 return good[0] if len(good)==1 else None


