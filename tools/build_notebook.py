"""Builds notebooks/kaggle_run.ipynb from src/ (self-contained, no drift).  Run: python tools/build_notebook.py"""
from pathlib import Path

import sys

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbsteps import cells as step_cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Potassium (Hyperkalemia) from the 12-lead ECG — MIMIC-IV-ECG + MIMIC-IV (Kaggle)

**Before Run All**
1. Settings → Internet **On** · keep this notebook **Private**. Accelerator: **None** while waveforms download
   (hours, no GPU needed — saves your GPU quota), then **GPU T4 x2** for training.
2. Add-ons → Google Cloud Services (BigQuery + Cloud Storage) or Google Cloud SDK — same as the other projects.
3. Check `project_id` in the config cell. The private bucket `<project_id>-mimic-ckpt` is reused/created automatically.

**What happens (everything resumable — if the session dies, just Run All again with the same `run_name`):**
1. Labels: every ECG with a potassium result within ±60 min (BigQuery, one query, cached).
2. Waveforms: all labelled ECGs are downloaded from the open MIMIC-IV-ECG mirror, resampled to 250 Hz and written
   as float16 shards to your bucket (finished shards are skipped on re-run).
3. Training: 1D ResNet on 2×T4 with mixed precision; full state checkpointed every 10 min + every epoch;
   stops cleanly at 11.3 h.
4. Test evaluation, machine-measurement baseline, population Grad-CAM, `ecg_k_resnet1d.pth`, GitHub report."""

INSTALL = "!pip install -q wfdb google-cloud-bigquery-storage db-dtypes 2>&1 | tail -1"

CONFIG = """import sys
sys.path.insert(0, "/kaggle/working/repo")
from src.config import Config

cfg = Config(
    project_id="mimic-cxr-training",   # <-- your GCP project ID (same as the other projects)
    bucket="",                         # empty = "<project_id>-mimic-ckpt" (private)
    run_name="main",                   # same name = resume
    max_ecgs=None,                     # None = FULL dataset; e.g. 20000 for a quick trial (use another run_name)
    epochs=25, batch_size=256,
    zip_parallel=6,                    # parallel ZIP range downloads (~0.3 MB/s each from physionet.org)
)
# TIP: the download phase needs NO GPU. Run it with Accelerator = None (saves GPU quota); when the log says
# "shards ready", switch to GPU T4 x2 and Run All again -> it goes straight to training."""

CHECK = """# pre-flight: everything should say OK
from src.gcp import check_access
from src.waveforms import check_source
check_access(cfg.project_id, ["mimiciv_hosp.labevents"])
check_source(cfg)
import torch; print("  GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"""


SHOW = """# Report (aggregate only -> safe for GitHub) + model file
import re, shutil, glob
from IPython.display import Markdown, Image, display
rep = f"{cfg.work_dir}/report"
md = open(f"{rep}/REPORT.md", encoding="utf-8").read()
display(Markdown(re.sub(r"!\\[.*?\\]\\(figures/.*?\\)", "", md)))
for f in sorted(glob.glob(f"{rep}/figures/*.png")):
    display(Image(f))
shutil.make_archive("/kaggle/working/results_for_github", "zip", rep)
print("Results zip  : /kaggle/working/results_for_github.zip  -> unzip into the repo as results/")
print("Model (.pth) :", f"{cfg.work_dir}/model/ecg_k_resnet1d.pth", "(also in your bucket under runs/main/results/model/)")"""


def main():
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_markdown_cell("## 1 · Setup"), nbf.v4.new_code_cell(INSTALL),
             nbf.v4.new_markdown_cell("## 2 · Project code\nThese cells write the package to `/kaggle/working/repo`. "
                                      "No need to edit them; the steps further down call this code."),
             nbf.v4.new_code_cell("!mkdir -p /kaggle/working/repo/src")]
    for f in sorted((ROOT / "src").glob("*.py")):
        cells.append(nbf.v4.new_code_cell(f"%%writefile /kaggle/working/repo/src/{f.name}\n" + f.read_text()))
    cells += [nbf.v4.new_markdown_cell("## 3 · Config\nChange only this cell."), nbf.v4.new_code_cell(CONFIG)]
    cells += [nbf.v4.new_markdown_cell("### Pre-flight check\nEverything should say OK."), nbf.v4.new_code_cell(CHECK)]
    steps, n = step_cells(ROOT / "src" / "pipeline.py", 4)
    cells += steps + [nbf.v4.new_markdown_cell(f"## {n} · Results report\nAggregate results only (safe for GitHub)."),
                      nbf.v4.new_code_cell(SHOW)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    out = ROOT / "notebooks" / "kaggle_run.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
