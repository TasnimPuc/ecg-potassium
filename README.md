# ECG → Potassium: detecting hyperkalemia from the 12-lead ECG

Predicts serum potassium (K⁺) **from the 12-lead ECG alone**, with no blood test. The model estimates
the K⁺ value and the probability of hyperkalemia (K ≥ 5.5, and ≥ 6.0) and hypokalemia (K < 3.5).
A population Grad-CAM shows **where in the heartbeat the model looks**. Hyperkalemia classically peaks
the T-wave, so we measure how much of the attention falls in the ST–T segment compared with the QRS complex.

**Data**
- Every diagnostic ECG in **MIMIC-IV-ECG** (open access) that has a potassium result in **MIMIC-IV**
  `labevents` (credentialed) within **±60 min**.
- Serum potassium (itemid 50971) and blood-gas potassium (50822) are both used.
- Hemolyzed samples are excluded, because hemolysis falsely raises K⁺.
- Train/val/test split by **patient**.

**Pipeline**
1. **Labels:** one BigQuery query, then a nearest-in-time match per ECG (`merge_asof`), cached in your bucket.
2. **Waveforms:**
   - all labelled ECGs are pulled in parallel from the open AWS mirror (`physionet-open`), with physionet.org as fallback;
   - WFDB files are parsed natively (format 16), and leads are put in a fixed order;
   - signals are resampled 500 → 250 Hz and written as float16 shards of 4,096 ECGs to your private GCS bucket.
3. **Model:** a 1D ResNet (12 × 2500 input) with three heads (K⁺ regression with Huber loss, hyper- and
   hypokalemia classification with BCE).
   - Augmentation: amplitude scaling, noise, lead dropout and time shift.
   - Training uses AMP and runs on 2×T4 via DataParallel, with cosine LR and early stopping on val AUROC.
4. **Evaluation:**
   - AUROC/AUPRC with bootstrap 95% CI, sensitivity at 90% specificity, and MAE / Pearson r;
   - sex and age subgroups;
   - a **baseline** XGBoost model on the ECG machine's own intervals and axes (QTc, QRS, T-axis…);
   - an R-peak-aligned population Grad-CAM.

**Resume anywhere:** shards, labels and the full training state are checkpointed to your bucket.
- Training state (model, optimizer, scheduler, AMP scaler, epoch, shard position, early-stopping counters)
  is saved every 10 min and at every epoch end.
- A run stops cleanly at 11.3 h.
- **Run All again with the same `run_name`** to continue. A crash loses at most 10 minutes of training.

## Run on Kaggle
1. Import `notebooks/kaggle_run.ipynb`. Set **GPU T4 x2**, Internet **On**, **Private**.
2. Add-ons → Google Cloud Services (BigQuery + Cloud Storage), the same setup as the other projects.
3. Set `project_id`, then Run All. The pre-flight cell must show BigQuery, GCS, `labevents` and the ECG source as OK.
4. When it finishes, download `results_for_github.zip` and unzip it here as `results/`.

## Model file (`.pth`)
`ecg_k_resnet1d.pth` is a bundle containing the weights, architecture, label normalisation, sampling
rate, lead order and thresholds. It is saved to `/kaggle/working/ecg_potassium/model/` and to
`gs://<bucket>/ecg-potassium/runs/<run_name>/results/model/`.

```bash
python -m src.predict --weights ecg_k_resnet1d.pth --record path/to/40689238   # WFDB .hea/.dat
```

> **Do not publish the weights on GitHub/Hugging Face.** They were trained on credentialed MIMIC-IV labels.
> PhysioNet asks for models derived from MIMIC to be shared on PhysioNet under the same agreement.
> `.gitignore` blocks `*.pth`.

## Results
See **[results/REPORT.md](results/REPORT.md)**. It contains aggregate numbers only, with cells under 10 suppressed.

## Local / CI
```bash
pip install -r requirements.txt
python -m pytest -q             # synthetic ECGs: parser, model, Grad-CAM, crash→resume, .pth inference
python tools/build_notebook.py  # regenerate the Kaggle notebook after editing src/
```

## Data use
Code only. MIMIC-IV-ECG is open access (ODbL). Potassium labels need your own PhysioNet credentialed
access to MIMIC-IV. Never commit data, shards, predictions or weights.
