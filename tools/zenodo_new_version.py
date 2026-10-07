"""Publish a new version of the Zenodo record for this repository.

Usage (from the repository root, after `git tag vX.Y.Z` and `git push --tags`):
    set ZENODO_TOKEN=...            # personal access token with deposit:write + deposit:actions
    python tools/zenodo_new_version.py vX.Y.Z

Creates a `git archive` zip of the tag, opens a new version of record 23216193 (concept record of the
manual upload of v0.1.0), replaces the file, sets the version, and publishes. Prints the new DOI.
"""
import os, sys, json, subprocess, urllib.request, urllib.error
RECORD = "23216193"
API = "https://zenodo.org/api"
tok = os.environ.get("ZENODO_TOKEN")
if not tok or len(sys.argv) < 2: sys.exit(__doc__)
tag = sys.argv[1]; ver = tag.lstrip("v")
zipname = f"llm-eval-high-stakes-{tag}.zip"
subprocess.run(["git", "archive", "--format=zip", f"--prefix=llm-eval-high-stakes-{tag}/", "-o", zipname, tag], check=True)

def req(method, url, data=None, headers=None, raw=False):
    h = {"Authorization": f"Bearer {tok}"}; h.update(headers or {})
    body = data if raw else (json.dumps(data).encode() if data is not None else None)
    if not raw and data is not None: h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(r) as resp:
            txt = resp.read().decode(); return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} -> {e.code}: {e.read().decode()[:500]}")

dep = req("GET", f"{API}/deposit/depositions/{RECORD}")
nv = req("POST", f"{API}/deposit/depositions/{RECORD}/actions/newversion")
draft_url = nv["links"]["latest_draft"]; draft = req("GET", draft_url)
for f in draft.get("files", []): req("DELETE", f["links"]["self"])
bucket = draft["links"]["bucket"]
with open(zipname, "rb") as fh:
    req("PUT", f"{bucket}/{zipname}", data=fh.read(), headers={"Content-Type": "application/octet-stream"}, raw=True)
meta = draft["metadata"]; meta["version"] = ver
meta.pop("doi", None); meta.pop("prereserve_doi", None)
req("PUT", draft_url, {"metadata": meta})
pub = req("POST", draft["links"]["publish"])
print("published:", pub.get("doi"), pub.get("links", {}).get("record_html"))
