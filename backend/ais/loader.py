import pandas as pd
from .validation import validate
def load(path):
 df=pd.read_csv(path);r=validate(df)
 if not r['valid']: raise ValueError('Invalid AIS CSV: missing required columns '+str(r['missing_required']))
 return r['data'],r['errors']
