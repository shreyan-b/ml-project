"""
Patch ensemble_voting_heart.ipynb so it opens in JupyterLab / Notebook
as well as Google Colab.

Changes made:
  - nbformat_minor : 0  →  5   (modern standard; required by JupyterLab)
  - kernelspec.display_name : "Python 3"  →  "Python 3 (ipykernel)"
  - kernelspec.language     : (missing)   →  "python"
  - language_info           : { name only } → full descriptor block
  - colab.toc_visible       : (missing)   →  true
  - Each cell gains an "id" tag unique to Jupyter (nbformat ≥ 4.5 requirement)
"""

import json, uuid, pathlib

NB = pathlib.Path(__file__).with_name("ensemble_voting_heart.ipynb")

nb = json.loads(NB.read_text(encoding="utf-8"))

# ── 1. Top-level format ────────────────────────────────────────────────────
nb["nbformat_minor"] = 5

# ── 2. kernelspec ──────────────────────────────────────────────────────────
nb["metadata"]["kernelspec"] = {
    "name"        : "python3",
    "display_name": "Python 3 (ipykernel)",
    "language"    : "python",
}

# ── 3. language_info ───────────────────────────────────────────────────────
nb["metadata"]["language_info"] = {
    "name"             : "python",
    "version"          : "3.10.0",
    "mimetype"         : "text/x-python",
    "file_extension"   : ".py",
    "pygments_lexer"   : "ipython3",
    "codemirror_mode"  : {"name": "ipython", "version": 3},
    "nbconvert_exporter": "python",
}

# ── 4. Colab extra field ───────────────────────────────────────────────────
nb["metadata"].setdefault("colab", {})["toc_visible"] = True

# ── 5. Cell IDs  (nbformat 4.5+ requires each cell to have a unique "id") ─
for cell in nb.get("cells", []):
    if "id" not in cell:
        cell["id"] = uuid.uuid4().hex[:8]

# ── 6. Write back ──────────────────────────────────────────────────────────
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[OK] Patched: {NB.name}")
print(f"     nbformat_minor   -> {nb['nbformat_minor']}")
print(f"     kernelspec       -> {nb['metadata']['kernelspec']}")
print(f"     language_info    -> {nb['metadata']['language_info']}")
print(f"     cells with id    -> {sum(1 for c in nb['cells'] if 'id' in c)} / {len(nb['cells'])}")
