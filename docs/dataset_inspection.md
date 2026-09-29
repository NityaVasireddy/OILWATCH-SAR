# Dataset inspection

- DATASET_ROOT: `.`
- Exists: `True`

## Summary
```json
{
  "file_count": 113,
  "extension_counts": {
    "": 3,
    ".example": 1,
    ".md": 8,
    ".txt": 1,
    ".py": 32,
    ".yaml": 2,
    ".html": 1,
    ".json": 1,
    ".jsx": 14,
    ".js": 1,
    ".css": 1,
    ".csv": 2,
    ".tif": 42,
    ".tmp": 1,
    ".pyc": 3
  },
  "image_like_count": 42
}
```
## Archive / metadata notes

## Sample file inspection
| Path | Size | Shape | Channels | Dtype | Min | Max | CRS | Resolution |
|---|---:|---|---:|---|---:|---:|---|---|
| `.env` | 87 | × |  |  |  |  |  |  |
| `.env.example` | 87 | × |  |  |  |  |  |  |
| `.gitignore` | 96 | × |  |  |  |  |  |  |
| `README.md` | 5342 | × |  |  |  |  |  |  |
| `requirements.txt` | 287 | × |  |  |  |  |  |  |
| `backend\main.py` | 1536 | × |  |  |  |  |  |  |
| `backend\__init__.py` | 0 | × |  |  |  |  |  |  |
| `configs\preprocessing.yaml` | 129 | × |  |  |  |  |  |  |
| `configs\training.yaml` | 135 | × |  |  |  |  |  |  |
| `docs\ais_methodology.md` | 367 | × |  |  |  |  |  |  |
| `docs\architecture.md` | 323 | × |  |  |  |  |  |  |
| `docs\dataset.md` | 319 | × |  |  |  |  |  |  |
| `docs\drift_methodology.md` | 431 | × |  |  |  |  |  |  |
| `docs\limitations.md` | 499 | × |  |  |  |  |  |  |
| `docs\model.md` | 276 | × |  |  |  |  |  |  |
| `docs\research.md` | 508 | × |  |  |  |  |  |  |
| `frontend\index.html` | 72 | × |  |  |  |  |  |  |
| `frontend\package.json` | 258 | × |  |  |  |  |  |  |
| `tests\test_api_contract.py` | 805 | × |  |  |  |  |  |  |
| `tests\test_core.py` | 483 | × |  |  |  |  |  |  |
| `frontend\src\App.jsx` | 7818 | × |  |  |  |  |  |  |
| `frontend\src\main.jsx` | 178 | × |  |  |  |  |  |  |
| `frontend\src\components\AisMap.jsx` | 709 | × |  |  |  |  |  |  |
| `frontend\src\components\AisUpload.jsx` | 200 | × |  |  |  |  |  |  |
| `frontend\src\components\DetectionResult.jsx` | 463 | × |  |  |  |  |  |  |
| `frontend\src\components\DriftPanel.jsx` | 214 | × |  |  |  |  |  |  |
| `frontend\src\components\Header.jsx` | 316 | × |  |  |  |  |  |  |
| `frontend\src\components\ModelStatus.jsx` | 268 | × |  |  |  |  |  |  |
| `frontend\src\components\ReportExport.jsx` | 146 | × |  |  |  |  |  |  |
| `frontend\src\components\SarUpload.jsx` | 321 | × |  |  |  |  |  |  |
| `frontend\src\components\SarViewer.jsx` | 140 | × |  |  |  |  |  |  |
| `frontend\src\components\Timeline.jsx` | 195 | × |  |  |  |  |  |  |
| `frontend\src\components\VesselDetails.jsx` | 278 | × |  |  |  |  |  |  |
| `frontend\src\components\VesselTable.jsx` | 339 | × |  |  |  |  |  |  |
| `frontend\src\services\api.js` | 949 | × |  |  |  |  |  |  |
| `frontend\src\styles\app.css` | 1687 | × |  |  |  |  |  |  |
| `dataset\raw\train\dataframe_train_dataset_256_90.csv` | 1647567 | × |  |  |  |  |  |  |
| `dataset\raw\train\dataframe_val_dataset_256_90.csv` | 549041 | × |  |  |  |  |  |  |
| `dataset\raw\train\images\2018_08_21_.tif` | 104296988 | 5701×4572 | 1 | ('float32',) | -40.58983612060547 | 18.38199806213379 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\2018_09_14_.tif` | 46362380 | 3602×3216 | 1 | ('float32',) | -39.627342224121094 | 17.212121963500977 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\2018_12_07.tif` | 39993899 | 4424×2259 | 1 | ('float32',) | -32.720298767089844 | 13.832046508789062 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\2018_12_07_b.tif` | 40534931 | 4922×2058 | 1 | ('float32',) | -23.498327255249023 | 15.48961067199707 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\2018_12_19.tif` | 50028107 | 5164×2421 | 1 | ('float32',) | -32.086483001708984 | 20.04606819152832 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\2018_12_19_b.tif` | 40200283 | 4640×2165 | 1 | ('float32',) | -30.585346221923828 | 27.417783737182617 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\2018_12_31_b.tif` | 66013323 | 5564×2965 | 1 | ('float32',) | -34.777061462402344 | 26.264524459838867 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\20190816.tif` | 26971059 | 3470×1942 | 1 | ('float32',) | -35.541507720947266 | 16.422046661376953 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\20190908.tif` | 24848303 | 3283×1891 | 1 | ('float32',) | -38.46342849731445 | 17.8741512298584 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\20200224.tif` | 52186363 | 3108×4195 | 1 | ('float32',) | -37.66067886352539 | 15.890119552612305 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\20200307.tif` | 26042203 | 3263×1994 | 1 | ('float32',) | -32.31134033203125 | 16.44980812072754 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\20200319.tif` | 12245459 | 2072×1476 | 1 | ('float32',) | -32.413612365722656 | 9.235625267028809 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\20200331.tif` | 29639939 | 3690×2007 | 1 | ('float32',) | -33.11437225341797 | 15.81337833404541 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\images\20200822.tif` | 21921443 | 2978×1839 | 1 | ('float32',) | -32.70995330810547 | 15.669501304626465 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\2018_08_21_.tif` | 104296836 | 5701×4572 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\2018_09_14_.tif` | 46362228 | 3602×3216 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\2018_12_07.tif` | 39993708 | 4424×2259 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\2018_12_07_b.tif` | 40534740 | 4922×2058 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\2018_12_19.tif` | 50027916 | 5164×2421 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\2018_12_19_b.tif` | 40200092 | 4640×2165 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\2018_12_31_b.tif` | 66013132 | 5564×2965 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\20190816.tif` | 26970868 | 3470×1942 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\20190908.tif` | 24848112 | 3283×1891 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\20200224.tif` | 52186172 | 3108×4195 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\20200307.tif` | 26042012 | 3263×1994 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\20200319.tif` | 12245268 | 2072×1476 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\20200331.tif` | 29639748 | 3690×2007 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\train\masks\20200822.tif` | 21921252 | 2978×1839 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\images\2018_09_26.tif` | 51969263 | 5083×2555 | 1 | ('float32',) | -36.45032501220703 | 20.961740493774414 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\images\2018_12_19_d.tif` | 10455251 | 2340×1116 | 1 | ('float32',) | -34.257938385009766 | 12.130539894104004 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\images\2018_12_19_e.tif` | 35929043 | 3772×2380 | 1 | ('float32',) | -34.47761535644531 | 18.722408294677734 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\images\2018_12_19_f_.tif` | 33222956 | 3250×2554 | 1 | ('float32',) | -37.574031829833984 | 14.017058372497559 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\images\20191015.tif` | 34010627 | 5050×1683 | 1 | ('float32',) | -34.675025939941406 | 20.585819244384766 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\images\20200224_b.tif` | 17719475 | 2590×1709 | 1 | ('float32',) | -27.328933715820312 | 16.86005973815918 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\images\20200319b.tif` | 16154579 | 2641×1528 | 1 | ('float32',) | -32.04018783569336 | 15.829171180725098 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\masks\2018_09_26.tif` | 51969072 | 5083×2555 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\masks\2018_12_19_d.tif` | 10455060 | 2340×1116 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\masks\2018_12_19_e.tif` | 35928852 | 3772×2380 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\masks\2018_12_19_f_.tif` | 33222804 | 3250×2554 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\masks\20191015.tif` | 34010436 | 5050×1683 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\masks\20200224_b.tif` | 17719284 | 2590×1709 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `dataset\raw\test\masks\20200319b.tif` | 16154388 | 2641×1528 | 1 | ('float32',) | 0.0 | 1.0 | EPSG:32616 | [10.0, 10.0] |
| `backend\ais\correlation.py` | 2859 | × |  |  |  |  |  |  |
| `backend\ais\loader.py` | 246 | × |  |  |  |  |  |  |
| `backend\ais\trajectory.py` | 100 | × |  |  |  |  |  |  |
| `backend\ais\validation.py` | 922 | × |  |  |  |  |  |  |
| `backend\ais\__init__.py` | 0 | × |  |  |  |  |  |  |
| `backend\api\ais.py` | 672 | × |  |  |  |  |  |  |
| `backend\api\correlation.py` | 1591 | × |  |  |  |  |  |  |
| `backend\api\detection.py` | 2254 | × |  |  |  |  |  |  |
| `backend\api\drift.py` | 520 | × |  |  |  |  |  |  |
| `backend\api\report.py` | 444 | × |  |  |  |  |  |  |
| `backend\api\__init__.py` | 0 | × |  |  |  |  |  |  |
| `backend\data\inspect_dataset.py` | 5234 | × |  |  |  |  |  |  |
| `backend\data\inspect_dataset.py.tmp` | 0 | × |  |  |  |  |  |  |
| `backend\data\prepare_dataset.py` | 1998 | × |  |  |  |  |  |  |
| `backend\data\split_dataset.py` | 708 | × |  |  |  |  |  |  |
| `backend\data\__init__.py` | 0 | × |  |  |  |  |  |  |
| `backend\geospatial\drift.py` | 1097 | × |  |  |  |  |  |  |
| `backend\geospatial\raster.py` | 513 | × |  |  |  |  |  |  |
| `backend\geospatial\spill_analysis.py` | 2097 | × |  |  |  |  |  |  |
| `backend\geospatial\__init__.py` | 0 | × |  |  |  |  |  |  |