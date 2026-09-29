# OILWATCH-SAR

AI-assisted maritime oil-spill investigation prototype for Smart India Hackathon 2026, SIH26143.

## What is implemented

- FastAPI backend
- Dataset discovery and inspection without assuming archive structure
- Image/mask pairing by normalized identifiers
- SHA-256 duplicate detection
- Scene-aware train/validation/test manifest generation
- One-channel SAR preprocessing for Sentinel-1A VV
- PyTorch U-Net segmentation
- BCE + Dice loss
- Reproducible training with early stopping, checkpointing, LR scheduling and CUDA AMP
- Real evaluation metrics and prediction visualizations
- Real SAR inference endpoint
- Spill geometry and physical-area calculation when raster resolution exists
- Simplified vector drift backtracking
- AIS CSV validation and trajectory construction
- Data-driven vessel correlation endpoint
- React/Vite operator dashboard
- Human-validation state and JSON report export
- Fail-closed real mode: no model/data means no fabricated evidence

## Important status

The codebase is complete as a prototype implementation, but a trained model cannot be included honestly until the real training dataset has been downloaded and processed. The repository does **not** contain fabricated model weights, metrics, AIS records, coordinates or vessel conclusions.

The official Zenodo record lists `Radar_data.rar` at 487.6 MB and describes 23 Gulf of Mexico oil-spill scenes using Sentinel-1A GRD VV:

https://zenodo.org/records/4672426

## Windows setup

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set the actual extracted dataset directory in `.env`:

```text
DATASET_ROOT=C:\path\to\Radar_data
MODEL_PATH=backend/model/best_model.pth
API_HOST=127.0.0.1
API_PORT=8000
```

## Required real-data sequence

Run from the repository root:

```powershell
python -m backend.data.inspect_dataset
python -m backend.data.prepare_dataset
python -m backend.data.split_dataset
python -m backend.ml.train
python -m backend.ml.evaluate
```

The inspection stage writes:

```text
dataset/metadata/dataset_inspection.json
dataset/metadata/duplicates.csv
docs/dataset_inspection.md
```

Preparation writes:

```text
dataset/metadata/pairing_report.csv
dataset/metadata/duplicates.csv
```

Splitting writes:

```text
dataset/metadata/split_manifest.csv
```

Training writes only after actual training:

```text
backend/model/best_model.pth
backend/model/model_metadata.json
```

Evaluation writes only after actual test inference:

```text
results/metrics.json
results/predictions/
```

## Start backend

```powershell
uvicorn backend.main:app --reload --port 8000
```

Health:

```text
http://127.0.0.1:8000/api/health
```

## Start frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Real application workflow

1. Upload a real Sentinel-1 SAR image.
2. Backend validates and preprocesses it.
3. Trained U-Net produces a pixel-level probability mask.
4. Spill geometry is calculated from the actual mask.
5. Geographic coordinates are returned only when raster geospatial metadata exists.
6. Environmental inputs can be supplied to the simplified drift model.
7. Upload a real AIS CSV.
8. AIS records are validated and grouped into vessel trajectories.
9. The correlation endpoint derives vessel candidates from the actual AIS records and release region.
10. Results are presented as investigation leads.
11. Export the analysis report.

## API

```text
GET  /api/health
POST /api/detect
POST /api/drift
POST /api/ais/upload
POST /api/correlate
POST /api/report
```

## Mandatory fail-closed behavior

- No SAR -> no detection.
- Valid SAR + no model -> HTTP 503 with `MODEL NOT LOADED — TRAIN THE MODEL OR PROVIDE MODEL WEIGHTS.`
- No AIS -> no vessel correlation.
- No geospatial metadata -> no fabricated latitude/longitude.
- No valid AIS coordinates/timestamps -> no vessel candidate.

## Testing

```powershell
python -m pytest -q
python -m compileall -q backend
```

The repository was checked in the build environment with **8/8 automated backend tests passing** and Python compilation succeeding.

The frontend dependency installation/build was not validated in the build environment because `npm install` exceeded the available execution window. Do not treat that as a successful frontend build until it is run locally.

## Scientific limitations

- SAR dark spots have non-oil look-alikes.
- The training dataset is small and domain-specific.
- Segmentation uncertainty propagates into geometry and drift.
- The drift model is deliberately simplified and assumes constant vectors.
- AIS contains gaps/errors and can be spoofed or unavailable.
- Geographic coordinates require valid SAR georeferencing.
- Release time is uncertain unless independently supplied.
- Correlation is an investigation lead, not proof of responsibility.

The application must never claim 100% accuracy, identify a confirmed polluter, or provide an exact release point.

## Scientific references

- Ramirez, W. A. Oil Spill Segmentation dataset, Zenodo, DOI 10.5281/zenodo.4672426.
- European Space Agency, Sentinel-1 mission: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1
- NOAA oil-spill resources: https://response.restoration.noaa.gov/oil-and-chemical-spills/oil-spills
