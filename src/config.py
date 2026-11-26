from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # --- GCP (only project_id is required) ---
    project_id: str = ""
    bucket: str = ""                     # empty = auto-create private "<project_id>-mimic-ckpt"
    bucket_location: str = "US"
    gcs_prefix: str = "ecg-potassium"
    run_name: str = "main"               # same name = resume
    local_store: str = ""                # tests only

    # --- data sources ---
    hosp_dataset: str = ""               # empty = auto-detect (potassium labs are credentialed MIMIC-IV)
    hosp_candidates: list = field(default_factory=lambda: ["mimiciv_v3_1_hosp", "mimiciv_hosp", "mimiciv_3_1_hosp", "mimiciv_v2_2_hosp"])
    # MIMIC-IV-ECG is open access: public AWS mirror first, PhysioNet as fallback (no login needed)
    wave_base_urls: list = field(default_factory=lambda: [
        "https://physionet-open.s3.amazonaws.com/mimic-iv-ecg/1.0/",
        "https://physionet-open.s3.us-east-1.amazonaws.com/mimic-iv-ecg/1.0/",
        "https://s3.amazonaws.com/physionet-open/mimic-iv-ecg/1.0/",
        "https://physionet.org/files/mimic-iv-ecg/1.0/"])
    s3_bucket_url: str = "https://physionet-open.s3.amazonaws.com/"   # public AWS Open Data bucket (auto-discovers the prefix)
    s3_dataset: str = "mimic-iv-ecg"
    wave_source: str = "zip"             # zip = PhysioNet ZIP via HTTP Range (fast, few connections) | files | auto
    zip_url: str = ""                    # empty = read the link from the PhysioNet project page
    zip_chunk_mb: float = 256            # one checkpointed pack per ~256 MB of the ZIP
    zip_parallel: int = 6                # parallel range downloads (physionet.org gives ~0.3 MB/s per connection)
    physionet_org_workers: int = 4       # physionet.org throttles heavy parallel downloads
    max_fail_frac: float = 0.02          # a shard with more failures is NOT committed (retried next run)

    # --- labels ---
    k_itemids: tuple = (50971, 50822)    # serum potassium (chemistry) + whole-blood potassium (blood gas)
    window_min: int = 60                 # potassium drawn within +-60 min of the ECG
    exclude_hemolyzed: bool = True       # hemolysis falsely raises K+
    k_min: float = 1.5
    k_max: float = 10.0
    hyper_thr: float = 5.5
    severe_thr: float = 6.0
    hypo_thr: float = 3.5
    max_ecgs: int | None = None          # None = FULL dataset (every ECG with a matched K+)
    split: tuple = (0.8, 0.1, 0.1)       # patient-level train/val/test
    seed: int = 42

    # --- waveform preprocessing (done once, sharded into the bucket, resumable) ---
    fs_out: int = 250                    # 500 Hz -> 250 Hz
    shard_size: int = 4096               # ECGs per shard (~245 MB float16)
    download_workers: int = 64
    cache_dir: str = "/tmp/ecgk_cache"   # local shard cache (auto-evicted when disk is low)
    min_free_gb: float = 8.0

    # --- model / training ---
    width: int = 64
    kernel: int = 7
    dropout: float = 0.3
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_frac: float = 0.03
    patience: int = 5                    # early stopping on val AUROC (K >= hyper_thr)
    amp: bool = True
    ckpt_minutes: float = 10.0           # full training state -> bucket at least this often
    max_runtime_hours: float = 11.3      # stop cleanly before Kaggle's 12 h limit

    # --- evaluation ---
    n_boot: int = 500
    gradcam_n: int = 2000                # ECGs per group used for the population Grad-CAM beat
    min_cell: int = 10

    # --- paths ---
    work_dir: str = "/kaggle/working/ecg_potassium"     # model + aggregate report
    private_dir: str = "/tmp/ecgk_private"              # labels / predictions (patient-level)
    cleanup_private: bool = True

    train_needs_gpu: bool = True         # CPU session: stop after downloading shards (no slow CPU training)

    # --- tests ---
    debug_crash_after_shards: int | None = None

    def to_dict(self):
        return asdict(self)
