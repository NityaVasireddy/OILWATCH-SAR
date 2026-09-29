from pathlib import Path
try: import rasterio
except Exception:rasterio=None

def metadata(path):
 if rasterio is None:return {'geospatial_available':False,'reason':'rasterio unavailable'}
 try:
  with rasterio.open(path) as d:return {'geospatial_available':bool(d.crs and d.transform),'crs':str(d.crs) if d.crs else None,'transform':list(d.transform),'bounds':list(d.bounds),'resolution':list(d.res),'width':d.width,'height':d.height}
 except Exception as e:return {'geospatial_available':False,'reason':str(e)}
