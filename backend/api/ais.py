from fastapi import APIRouter,UploadFile,File,HTTPException
import tempfile,os
from ..ais.loader import load
from ..ais.trajectory import trajectories
router=APIRouter()
@router.post('/ais/upload')
async def upload(file:UploadFile=File(...)):
 if not file.filename.lower().endswith('.csv'):raise HTTPException(400,'AIS input must be CSV.')
 fd,path=tempfile.mkstemp(suffix='.csv');os.close(fd)
 try:
  with open(path,'wb') as f:f.write(await file.read())
  df,errors=load(path);return {'rows':len(df),'vessels':df.MMSI.nunique(),'validation_warnings':errors,'trajectories':trajectories(df)}
 except ValueError as e:raise HTTPException(400,str(e))
 finally:os.unlink(path)
