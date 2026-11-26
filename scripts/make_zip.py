"""Bundle the project into dist/project.zip for uploading to Colab (no git push needed)."""

import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ["scalable_agents", "scripts", "tests", "notebooks", "requirements.txt", "README.md"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}

os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
out = os.path.join(ROOT, "dist", "project.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in INCLUDE:
        path = os.path.join(ROOT, item)
        if os.path.isfile(path):
            zf.write(path, item)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                full = os.path.join(dirpath, name)
                zf.write(full, os.path.relpath(full, ROOT))
print(f"wrote {out} ({os.path.getsize(out) / 1e3:.0f} kB)")
