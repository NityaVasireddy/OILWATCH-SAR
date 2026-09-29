from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile, os
import pandas as pd
from ..ais.loader import load
from ..ais.correlation import correlate_dataframe

router = APIRouter()


@router.post('/correlate')
async def correlate(
    file: UploadFile = File(...),
    release_lat: float = Form(...),
    release_lon: float = Form(...),
    release_time: str | None = Form(None),
):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(400, 'AIS input must be CSV.')
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        with open(path, 'wb') as f:
            f.write(await file.read())
        df, warnings = load(path)
        candidates = correlate_dataframe(df, release_lat, release_lon, release_time)
        if not candidates:
            raise HTTPException(422, 'No usable vessel trajectories found in AIS data.')
        return {
            'release_region': {'latitude': release_lat, 'longitude': release_lon},
            'release_time': release_time,
            'vessel_count': len(candidates),
            'validation_warnings': warnings,
            'candidates': candidates,
            'disclaimer': 'Investigation lead only — correlation does not establish responsibility.'
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f'Unable to correlate AIS data: {e}')
    finally:
        if os.path.exists(path):
            os.unlink(path)
