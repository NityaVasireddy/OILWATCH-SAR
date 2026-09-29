import pandas as pd
REQUIRED=['MMSI','timestamp','latitude','longitude','SOG','COG']
def validate(df):
 cols={c.lower():c for c in df.columns};missing=[c for c in REQUIRED if c.lower() not in cols]
 if missing:return {'valid':False,'missing_required':missing,'errors':[]}
 d=df.rename(columns={v:k for k,v in cols.items()}).copy(); errs=[]
 d['timestamp']=pd.to_datetime(d['timestamp'],errors='coerce',utc=True);d['latitude']=pd.to_numeric(d['latitude'],errors='coerce');d['longitude']=pd.to_numeric(d['longitude'],errors='coerce');d['SOG']=pd.to_numeric(d['SOG'],errors='coerce');d['COG']=pd.to_numeric(d['COG'],errors='coerce')
 bad=(~d.latitude.between(-90,90))|(~d.longitude.between(-180,180))|d.timestamp.isna();
 if bad.any():errs.append(f'{int(bad.sum())} invalid coordinate/timestamp rows')
 d=d[~bad].copy();d=d.sort_values(['MMSI','timestamp']);return {'valid':True,'missing_required':[],'errors':errs,'data':d}
