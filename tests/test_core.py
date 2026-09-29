import numpy as np
from backend.geospatial.spill_analysis import analyze
from backend.geospatial.drift import backtrack
from backend.ais.correlation import angle_diff,score

def test_geometry_no_spill(): assert analyze(np.zeros((4,4),bool))['detected_pixel_count']==0
def test_drift_moves(): assert len(backtrack(10,20,5,0,1,0,2)['drift_path'])>1
def test_heading_wrap(): assert angle_diff(359,1)==2
def test_correlation_components(): assert score(0,0,0,1)['correlation_score']==100
