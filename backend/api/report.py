from fastapi import APIRouter
from fastapi.responses import Response
import json,datetime
router=APIRouter()
@router.post('/report')
def report(payload:dict):
 payload=dict(payload);payload['generated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();return Response(content=json.dumps(payload,indent=2,default=str),media_type='application/json',headers={'Content-Disposition':'attachment; filename=oilwatch_analysis_report.json'})
