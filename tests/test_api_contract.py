from fastapi.testclient import TestClient
from backend.main import app


def test_health_contract():
    r = TestClient(app).get('/api/health')
    assert r.status_code == 200
    assert {'backend_status','model_loaded','device','input_channels','input_size'} <= set(r.json())


def test_detect_without_model_returns_503_or_bad_request():
    r = TestClient(app).post('/api/detect', files={'file': ('x.png', b'not-an-image', 'image/png')})
    assert r.status_code in (400, 503)


def test_correlate_requires_multipart_inputs():
    r = TestClient(app).post('/api/correlate')
    assert r.status_code == 422


def test_report_generates_timestamp():
    r = TestClient(app).post('/api/report', json={'human_validation':'Needs Review'})
    assert r.status_code == 200
    assert 'generated_at' in r.json()
