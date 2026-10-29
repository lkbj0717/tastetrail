from pathlib import Path
import hashlib, io, os, ssl, zipfile
import httpx

private = Path(os.environ.get('V11_PRIVATE', str(Path.home()/'.local/share/tastetrail/culture_v11')))
private.mkdir(parents=True, exist_ok=True, mode=0o700)
archive = private/'ml-1m.zip'
if not archive.exists():
    response = httpx.get('https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        timeout=120, trust_env=False, verify=ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt'))
    response.raise_for_status()
    archive.write_bytes(response.content)
    archive.chmod(0o600)
with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as z:
    for name in ['movies.dat', 'ratings.dat', 'README']:
        p = private/name
        if not p.exists():
            p.write_bytes(z.read('ml-1m/'+name)); p.chmod(0o600)
print({'download_bytes': archive.stat().st_size, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
