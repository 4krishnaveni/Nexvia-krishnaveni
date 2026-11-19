# AI-Assisted Chest X-ray Review System

> **FOR EDUCATION ONLY. NOT FOR CLINICAL USE.**

DICOM chest X-ray -> Orthanc -> inference worker (DenseNet121 classifier + MONAI U-Net + Grad-CAM) -> FastAPI -> React viewer where a radiologist approves or rejects the AI finding. Every decision goes to an append-only audit log. Dataset: **RSNA Pneumonia Detection Challenge** (real DICOMs with boxes). Versioned with **DVC**, packaged with **Docker**.

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | API: [docs/API.md](docs/API.md) | Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

**Honest notes**
* The brief says "3D U-Net". RSNA is 2-D, so this uses a **2-D MONAI U-Net on box-derived masks** (state this in your presentation). Stretch goal: MSD Task06 Lung with `spatial_dims=3`.
* Code review status: see "What is verified" at the bottom.

---

## 0. Prerequisites (install once)

| Tool | Version | Install |
|---|---|---|
| Git | any | https://git-scm.com/downloads |
| Python | 3.10-3.12 (3.11 best) | https://www.python.org/downloads/ (Windows: tick "Add to PATH") |
| Node.js | 20 LTS | https://nodejs.org (only for running the frontend without Docker) |
| Docker Desktop | recent | https://www.docker.com/products/docker-desktop (Windows: enable WSL2) |
| Kaggle account | free | https://www.kaggle.com -> Settings -> Create New API Token (downloads `kaggle.json`) |
| GPU | optional | Training on CPU is too slow for the full dataset. Use a **Kaggle Notebook** (free GPU, 30 h/week) or Colab for sections 5-6. Everything else runs on a normal laptop. |

Windows users: run the commands in **Git Bash** or **WSL2 Ubuntu**. PowerShell equivalents are noted where they differ.

## 1. Get the code and create the environment

```bash
git clone <your-repo-url> cxr-review-system && cd cxr-review-system
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1   Git Bash: source .venv/Scripts/activate

# PyTorch FIRST (pick one):
pip install torch torchvision                                        # Mac / Linux GPU (CUDA wheels from PyPI)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU only (laptops, CI)
# For a specific CUDA version use the selector at https://pytorch.org/get-started/locally/

pip install -r requirements.txt
cp .env.example .env                 # then edit every "change-me" value
```

## 2. 15-minute smoke test with synthetic data (do this first)

This proves your whole install works before you spend hours downloading 4 GB.

```bash
python scripts/make_synthetic_rsna.py --out data/synthetic --n 150
python -m src.data.preprocess --raw data/synthetic --out data/syn_processed --size 128
python -m src.train.train_cls --data data/syn_processed --epochs 3 --batch-size 16 --no-pretrained --workers 0 --out models/syn_classifier.pt --report reports/syn_cls.json
python -m src.train.train_seg --data data/syn_processed --epochs 3 --batch-size 8 --workers 0 --out models/syn_unet.pt --report reports/syn_seg.json
python -m src.eval.evaluate --data data/syn_processed --cls models/syn_classifier.pt --seg models/syn_unet.pt --out reports/syn_metrics.json
pytest -q
```
The numbers are meaningless (fake data). You only need: no errors, and PNGs in `reports/figures/`.

## 3. Real data: RSNA Pneumonia Detection

1. On Kaggle open the competition `rsna-pneumonia-detection-challenge` and click **Join / Accept rules** (downloads fail with 403 until you do).
2. Put `kaggle.json` in `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\`), then:
```bash
pip install kaggle
mkdir -p data/raw && cd data/raw
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d . && cd ../..
ls data/raw        # expect stage_2_train_images/  stage_2_train_labels.csv  stage_2_detailed_class_info.csv
```
3. Learn DICOM: `python -m src.data.dicom_utils data/raw/stage_2_train_images/<any>.dcm` prints Modality, PhotometricInterpretation, PixelSpacing, etc. Write 10 lines about what you saw in `docs/DICOM_NOTES.md` (deliverable evidence).
4. Preprocess a small subset first, then everything:
```bash
python -m src.data.preprocess --raw data/raw --out data/processed --size 256 --limit 1000   # dev subset
python -m src.eval.eda --data data/processed                                                # EDA figures
python -m src.data.preprocess --raw data/raw --out data/processed --size 256                # full (about 20-40 min)
```
This performs the **patient-level** 70/15/15 split, writes `meta.csv` (labels, sex, age, view), `boxes.csv` and the masks.

## 4. Version data and pipeline with DVC

```bash
dvc init
dvc remote add -d storage ../dvc-storage          # or a Google Drive / S3 remote (see dvc.org/doc/user-guide/data-management/remote-storage)
dvc add data/raw                                   # tracks the raw dataset (never commit images to Git)
git add data/raw.dvc data/.gitignore .dvc && git commit -m "chore: track raw data with DVC"
dvc repro                                          # runs preprocess -> train_cls -> train_seg -> evaluate from dvc.yaml
dvc push
```
Hyper-parameters live in `params.yaml`; changing one makes `dvc repro` rerun only the affected stages. Teammates restore everything with `dvc pull`.

## 5. Train on a free GPU (Kaggle Notebook)

1. kaggle.com -> Code -> New Notebook -> right panel: **Accelerator: GPU T4**, **Add Input** -> search the RSNA competition. Push your repo to GitHub first.
2. Cells:
```python
!git clone https://github.com/<you>/cxr-review-system.git && cd cxr-review-system && pip -q install monai pydicom grad-cam pynetdicom
%cd cxr-review-system
!python -m src.data.preprocess --raw /kaggle/input/rsna-pneumonia-detection-challenge --out /kaggle/working/processed --size 256
!python -m src.train.train_cls --data /kaggle/working/processed --epochs 20 --out /kaggle/working/classifier.pt --report /kaggle/working/cls_train.json
!python -m src.train.train_seg --data /kaggle/working/processed --epochs 25 --out /kaggle/working/unet.pt --report /kaggle/working/seg_train.json
!python -m src.eval.evaluate --data /kaggle/working/processed --cls /kaggle/working/classifier.pt --seg /kaggle/working/unet.pt --out /kaggle/working/metrics.json
```
3. Download `classifier.pt`, `unet.pt`, `metrics.json`, `cls_train.json` (Output tab). Put the `.pt` files in `./models/`, metrics in `./reports/`, then `dvc add models/classifier.pt models/unet.pt` and commit the `.dvc` files.
   Try `--backbone efficientnet_b0` too and keep both result rows for your ablation table.

## 6. Read the results honestly

`reports/metrics.json` has AUROC, AUPRC, sensitivity/specificity (threshold picked on validation), ECE, **subgroup AUROC by sex and age**, Dice on positives, and the **Grad-CAM pointing-game accuracy** (does the heat peak fall inside the radiologist box). Open `reports/figures/gradcam_*_hit/miss.png` and pick 2 hits and 2 misses for the slides. Copy the numbers into `docs/MODEL_CARD.md`. Expect AUROC around 0.85-0.92 and Dice around 0.4-0.6; report whatever you actually get.

## 7. Run the whole system (Docker, one command)

```bash
# models/classifier.pt and models/unet.pt must exist.
# No weights yet? set ALLOW_UNTRAINED=1 in .env to test the plumbing (the UI shows a warning; scores are random).
docker compose up --build
```
| URL | What |
|---|---|
| http://localhost:3000 | React viewer (log in with RADIOLOGIST_USER / PASSWORD from `.env`) |
| http://localhost:8000/docs | API docs |
| http://localhost:8042 | Orthanc UI (ORTHANC_USER / PASSWORD from `.env`) |

Push studies into Orthanc (new terminal, venv active, `.env` values exported):
```bash
export ORTHANC_USER=orthanc ORTHANC_PASSWORD=<yours>
python scripts/upload_to_orthanc.py data/raw/stage_2_train_images --limit 5     # REST upload
python scripts/send_cstore.py data/raw/stage_2_train_images --limit 3           # or real DICOM C-STORE on :4242
```
Within about 10 seconds (StableAge 5 s + poll 5 s) the studies appear in the viewer. Click one, toggle the heatmap/mask, **Approve** or **Reject**, then open the **Audit log** tab.
`docker compose logs -f worker` shows inference activity. `docker compose down -v` wipes all state.

## 8. Run without Docker for the services you are editing

Orthanc still runs in Docker (`docker compose up orthanc`). Then three terminals:
```bash
# T1 API
export DB_PATH=./data/app.db RESULTS_DIR=./results JWT_SECRET=dev-secret-dev-secret-dev-secret-12 INTERNAL_API_KEY=dev-key RADIOLOGIST_USER=radiologist RADIOLOGIST_PASSWORD=radpass
uvicorn backend.app.main:app --reload --port 8000
# T2 worker
export ORTHANC_URL=http://localhost:8042 ORTHANC_USER=orthanc ORTHANC_PASSWORD=<yours> API_URL=http://localhost:8000 INTERNAL_API_KEY=dev-key RESULTS_DIR=./results   # + ALLOW_UNTRAINED=1 if no weights
python -m src.service.worker
# T3 frontend
cd frontend && npm install && npm run dev        # http://localhost:5173
```
(PowerShell: `$env:NAME="value"` instead of `export NAME=value`.)

## 9. Tests and CI
```bash
pytest -q          # DICOM loader (MONOCHROME1), masks, model shapes, Grad-CAM, API + audit append-only, worker idempotency
ruff check .
```
`.github/workflows/ci.yml` runs both plus the frontend build on every push. Save the pytest output to `reports/pytest.txt` for the deliverable.

## 10. Git workflow that reviews well
Branches: `feat/dicom-data`, `feat/models`, `feat/gradcam`, `feat/orthanc-worker`, `feat/api`, `feat/viewer`, `feat/docker-ci`, `docs/final`. One PR each, conventional messages (`feat: add MONOCHROME1 handling`, `test: audit log is append-only`, `docs: model card`). Never commit `data/raw`, `.pt` files or `.env`.

## 11. Deliverables map
| Deliverable | Where / how |
|---|---|
| Working demo | Screen-record 3-5 min: upload to Orthanc -> worker log -> viewer overlay + confidence -> Approve -> Audit tab |
| Repo + README | This repo; fill results table below |
| Architecture + API docs | `docs/ARCHITECTURE.md` (export the Mermaid as PNG via https://mermaid.live), `docs/API.md`, `openapi.json` |
| Test + accuracy numbers | `reports/pytest.txt`, `reports/metrics.json` |
| 10-min talk | Problem 1 / Design 2 / Data+models 2 / Explainability+viewer 2 / Challenges+limits 1.5 / Results+next 1.5 |
| Reflection | What you owned, one thing that broke and how you fixed it, what you would do next |

| Model | AUROC | Sens | Spec | Dice (box masks) | Pointing game |
|---|---|---|---|---|---|
| DenseNet121 + U-Net | TODO | TODO | TODO | TODO | TODO |
| EfficientNet-B0 (ablation) | TODO | TODO | TODO | - | TODO |

## 12. Troubleshooting
| Symptom | Fix |
|---|---|
| Kaggle download 403 | Accept the competition rules on the website first |
| `RuntimeError ... is a view and is being modified inplace` | Already avoided (in-place ReLUs are disabled in `CXRClassifier`); keep using that wrapper |
| Everything predicted positive/negative | Check the threshold in the checkpoint, check class balance, and make sure val has both classes |
| Viewer shows no studies | `docker compose logs worker`; missing weights -> set ALLOW_UNTRAINED=1 or train; Orthanc password mismatch between `.env` and the running container (`docker compose down -v` then up) |
| CUDA out of memory | `--batch-size 16` |
| Images look inverted | MONOCHROME1 not handled; always go through `dicom_to_array` |
| Orthanc env vars ignored | Confirm your image tag supports `ORTHANC__*` variables (`orthancteam/orthanc`), or mount an `orthanc.json` instead |

## 13. Known limits (also in the model card)
Single dataset, no external validation, opacity is not specific to pneumonia, box-derived masks, calibration on one validation set, demo file endpoint is unauthenticated (de-identified data only), SQLite and polling are fine for a demo, not for production.

## What is verified
The code was written and syntax-checked in a sandbox with no access to PyPI/npm, so **none of the Python tests, the training scripts or the React build were executed there**; `docker-compose.yml` passed `docker compose config`. Section 2 (smoke test) and `pytest` are how you confirm it on your machine. Expect to fix small version-specific issues; paste any traceback to get a quick fix.
