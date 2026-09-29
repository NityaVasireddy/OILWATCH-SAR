# Architecture
FastAPI provides health, SAR detection, drift, AIS and report endpoints. PyTorch performs segmentation. Raster/geospatial modules isolate metadata and geometry. React/Vite/Leaflet form the operator UI. Real mode is intentionally fail-closed: absent inputs or model weights do not produce synthetic evidence.
