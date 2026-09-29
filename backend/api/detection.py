from fastapi import APIRouter, UploadFile, File, HTTPException

import numpy as np

from ..ml.inference import InferenceEngine
from ..geospatial.spill_analysis import analyze


router = APIRouter()

engine = None


def get_engine():
    global engine

    if engine is None:
        try:
            engine = InferenceEngine()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "MODEL NOT LOADED — TRAIN THE MODEL OR PROVIDE MODEL WEIGHTS. "
                    f"({exc})"
                ),
            )

    return engine


def get_geospatial_metadata(file_bytes):
    """
    Read geographic metadata from the uploaded raster.

    No coordinates are invented if the input does not contain
    geospatial metadata.
    """

    metadata = {
        "georeferenced": False,
        "crs": None,
        "transform": None,
        "bounds": None,
        "resolution": None,
    }

    try:
        import rasterio

        from rasterio.io import MemoryFile

        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:

                metadata["georeferenced"] = bool(
                    src.crs is not None
                    and src.transform is not None
                )

                if src.crs is not None:
                    metadata["crs"] = str(src.crs)

                if src.transform is not None:
                    metadata["transform"] = [
                        float(value)
                        for value in src.transform
                    ]

                if src.bounds is not None:
                    metadata["bounds"] = {
                        "left": float(src.bounds.left),
                        "bottom": float(src.bounds.bottom),
                        "right": float(src.bounds.right),
                        "top": float(src.bounds.top),
                    }

                if src.res is not None:
                    metadata["resolution"] = [
                        float(src.res[0]),
                        float(src.res[1]),
                    ]

    except Exception:
        # Detection can still run without geospatial metadata.
        pass

    return metadata


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Run the trained U-Net model on a real SAR image.

    The endpoint does not generate fake detection values.
    """

    # ---------------------------------------------------------
    # Validate filename
    # ---------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was provided.",
        )

    filename = file.filename.lower()

    allowed_extensions = (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    )

    if not filename.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Upload a TIFF, GeoTIFF, PNG, or JPEG image."
            ),
        )

    try:
        # -----------------------------------------------------
        # Read uploaded file
        # -----------------------------------------------------

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # -----------------------------------------------------
        # Read geospatial metadata
        # -----------------------------------------------------

        geo = get_geospatial_metadata(file_bytes)

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # InferenceEngine.predict() expects a FastAPI
        # UploadFile object because it internally accesses:
        #
        #     image.file.read()
        #
        # Therefore we reset the UploadFile and pass `file`,
        # NOT BytesIO(file_bytes).
        # -----------------------------------------------------

        await file.seek(0)

        result_data, preprocessing_info, inference_seconds = (
            get_engine().predict(file)
        )

        # -----------------------------------------------------
        # Validate model output
        # -----------------------------------------------------

        if not isinstance(result_data, dict):
            raise HTTPException(
                status_code=500,
                detail="Unexpected model inference output.",
            )

        if "mask" not in result_data:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Model output does not contain a prediction mask."
                ),
            )

        if "probability_map" not in result_data:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Model output does not contain a probability map."
                ),
            )

        # -----------------------------------------------------
        # Extract actual model results
        # -----------------------------------------------------

        mask = np.asarray(
            result_data["mask"],
            dtype=np.uint8,
        )

        probability_map = np.asarray(
            result_data["probability_map"],
            dtype=np.float32,
        )

        # Ensure binary mask.
        mask = (mask > 0).astype(np.uint8)

        # -----------------------------------------------------
        # Calculate actual prediction statistics
        # -----------------------------------------------------

        spill_pixels = int(mask.sum())

        total_pixels = int(mask.size)

        if total_pixels > 0:
            spill_fraction = (
                spill_pixels / total_pixels
            )
        else:
            spill_fraction = 0.0

        probability_min = float(
            probability_map.min()
        )

        probability_max = float(
            probability_map.max()
        )

        probability_mean = float(
            probability_map.mean()
        )

        probability_median = float(
            np.median(probability_map)
        )

        if spill_pixels > 0:
            spill_confidence = float(
                probability_map[mask == 1].mean()
            )
        else:
            spill_confidence = 0.0

        # -----------------------------------------------------
        # Spill geometry
        #
        # Only provide physical/geospatial information when
        # real raster resolution is available.
        # -----------------------------------------------------

        resolution = None

        if geo.get("georeferenced"):
            resolution = geo.get("resolution")

        geometry_result = analyze(
            mask,
            probability_map,
            resolution,
        )

        # -----------------------------------------------------
        # Final response
        # -----------------------------------------------------

        response = {
            "filename": file.filename,

            "detection": {
                "oil_spill_detected": bool(
                    spill_pixels > 0
                ),

                "threshold": float(
                    result_data.get(
                        "threshold",
                        0.5,
                    )
                ),

                "spill_pixels": spill_pixels,

                "total_pixels": total_pixels,

                "spill_fraction": float(
                    spill_fraction
                ),

                "spill_coverage_percent": float(
                    spill_fraction * 100.0
                ),
            },

            "model_confidence": {
                "spill_region_mean": float(
                    spill_confidence
                ),

                "minimum_probability": probability_min,

                "maximum_probability": probability_max,

                "mean_probability": probability_mean,

                "median_probability": probability_median,
            },

            "geometry": geometry_result,

            "preprocessing": preprocessing_info,

            "inference_time_seconds": float(
                inference_seconds
            ),

            "geospatial_metadata": geo,

            "coordinates_note": (
                "Geographic coordinates are available from "
                "the raster metadata."
                if geo.get("georeferenced")
                else
                "Map unavailable because geographic "
                "coordinates are unavailable for this input."
            ),

            "scientific_note": (
                "AI-assisted segmentation result. "
                "The prediction is decision-support output "
                "and requires human validation."
            ),

            "investigation_note": (
                "Detection does not establish vessel "
                "responsibility. Vessel correlation requires "
                "separate real AIS data and spatial-temporal "
                "analysis."
            ),
        }

        return response

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SAR detection failed: {exc}",
        )