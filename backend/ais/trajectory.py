def trajectories(df): return {str(m):g.to_dict('records') for m,g in df.groupby('MMSI',sort=False)}
