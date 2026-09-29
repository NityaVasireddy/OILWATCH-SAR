import React, { useEffect, useState } from 'react';
import { health, detect, ais, correlate, drift } from './services/api';

const fmt = (v, d = 3) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '—';

export default function App() {
  const [h, setH] = useState(null);
  const [sar, setSar] = useState(null);
  const [sarFile, setSarFile] = useState(null);
  const [aisFile, setAisFile] = useState(null);
  const [aisInfo, setAisInfo] = useState(null);
  const [corr, setCorr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');
  const [driftForm, setDriftForm] = useState({ wind_speed: '', wind_direction: '', current_speed: '', current_direction: '', backtracking_hours: '6' });
  const [driftResult, setDriftResult] = useState(null);

  useEffect(() => { health().then(setH).catch(() => setErr('Backend unavailable. Start FastAPI first.')); }, []);

  const runDetection = async () => {
    if (!sarFile) return;
    setBusy(true); setErr(''); setSar(null); setCorr(null);
    try { setSar(await detect(sarFile)); }
    catch (e) { setErr(e.response?.data?.detail || e.message); }
    finally { setBusy(false); }
  };

  const runAis = async () => {
    if (!aisFile) return;
    setErr('');
    try { setAisInfo(await ais(aisFile)); }
    catch (e) { setErr(e.response?.data?.detail || e.message); }
  };

  const runCorrelation = async () => {
    if (!aisFile || !sar?.centroid_geographic) return;
    setErr('');
    try {
      setCorr(await correlate(aisFile, {
        release_lat: sar.centroid_geographic.latitude,
        release_lon: sar.centroid_geographic.longitude,
      }));
    } catch (e) { setErr(e.response?.data?.detail || e.message); }
  };

  const runDrift = async () => {
    if (!sar?.centroid_geographic) { setErr('Drift requires geographic coordinates from the SAR input.'); return; }
    try {
      setDriftResult(await drift({
        lat: sar.centroid_geographic.latitude,
        lon: sar.centroid_geographic.longitude,
        ...Object.fromEntries(Object.entries(driftForm).map(([k,v]) => [k, Number(v)]))
      }));
    } catch (e) { setErr(e.response?.data?.detail || e.message); }
  };

  const exportReport = () => {
    const payload = {
      input_file: sar?.filename,
      sar_metadata: sar?.geospatial_metadata,
      detection: sar ? { detected: sar.detected_pixel_count > 0, coverage_percent: sar.coverage_percent } : null,
      model_confidence: sar?.model_confidence,
      spill_geometry: sar,
      drift: driftResult,
      ais: aisInfo ? { rows: aisInfo.rows, vessels: aisInfo.vessels, validation_warnings: aisInfo.validation_warnings } : null,
      candidate_vessels: corr?.candidates || [],
      limitations: ['SAR look-alikes', 'Segmentation uncertainty', 'Simplified drift assumptions', 'AIS gaps/spoofing limitations', 'Correlation does not establish responsibility'],
      human_validation: 'Needs Review',
      generated_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const a = document.createElement('a');
    a.href = url; a.download = 'oilwatch_analysis_report.json'; a.click(); URL.revokeObjectURL(url);
  };

  return <div className="app">
    <header>
      <div><div className="brand">OILWATCH-SAR</div><div className="sub">AI-Powered SAR Oil Spill Detection & Vessel Investigation</div></div>
      <div className="status"><strong>{h?.model_loaded ? 'MODEL READY' : 'MODEL NOT LOADED'}</strong><span>{h?.device || '—'} · {h?.input_channels ?? '—'} channel</span></div>
    </header>

    <main>
      <section className="hero">
        <div className="eyebrow">MARITIME INTELLIGENCE / REAL MODE</div>
        <h1>Satellite evidence into explainable investigation leads.</h1>
        <p>Real input only. The application does not fabricate detection masks, coordinates, vessels, scores, or metrics.</p>
        <label className="upload">UPLOAD SAR IMAGE<input type="file" accept="image/*,.tif,.tiff" onChange={e => setSarFile(e.target.files[0] || null)} /></label>
        {sarFile && <button onClick={runDetection} disabled={busy}>{busy ? 'ANALYZING…' : 'ANALYZE SAR'}</button>}
      </section>

      {err && <div className="error">{err}</div>}

      <section className="grid">
        <div className="card"><h3>MODEL STATUS</h3><b>{h?.model_loaded ? 'READY' : 'NOT LOADED'}</b><p>Version: {h?.model_version || '—'}</p><p>Channels: {h?.input_channels ?? '—'}</p><p>Input size: {h?.input_size ?? '—'}</p><p>Device: {h?.device || '—'}</p></div>
        <div className="card"><h3>DETECTION</h3>{sar ? <><div className="metric">{sar.detected_pixel_count}</div><p>Detected pixels</p><p>Coverage: {fmt(sar.coverage_percent)}%</p><p>Inference: {fmt(sar.inference_time_seconds)} s</p><p>{sar.area_note}</p><p>Area: {sar.physical_area ?? '—'} {sar.physical_area_units || ''}</p>{sar.model_confidence && <p>Mean model probability: {fmt(sar.model_confidence.mean_probability, 4)}</p>}</> : <p>Upload a Sentinel-1 SAR image to begin.</p>}</div>
        <div className="card"><h3>GEOSPATIAL EVIDENCE</h3>{sar?.centroid_geographic ? <><p>Centroid latitude: {fmt(sar.centroid_geographic.latitude, 6)}</p><p>Centroid longitude: {fmt(sar.centroid_geographic.longitude, 6)}</p><p>Orientation: {fmt(sar.orientation_degrees, 2)}°</p></> : <p>Map unavailable because geographic coordinates are unavailable for this input.</p>}</div>
      </section>

      <section className="wide card"><h3>DRIFT BACKTRACKING · SIMPLIFIED DRIFT MODEL</h3>
        <div className="formgrid">{[['wind_speed','Wind speed'],['wind_direction','Wind direction °'],['current_speed','Current speed'],['current_direction','Current direction °'],['backtracking_hours','Hours']].map(([k,l]) => <label key={k}>{l}<input value={driftForm[k]} onChange={e => setDriftForm({...driftForm,[k]:e.target.value})} placeholder="Required" /></label>)}</div>
        <button onClick={runDrift}>BACKTRACK</button>{driftResult && <div className="result"><p>Estimated release region: {fmt(driftResult.estimated_release_region.latitude,6)}, {fmt(driftResult.estimated_release_region.longitude,6)}</p><p>Window: {driftResult.time_window_hours} hours</p></div>}
      </section>

      <section className="wide card"><h3>AIS INVESTIGATION</h3>
        <label className="secondary">UPLOAD AIS CSV<input type="file" accept=".csv" onChange={e => { setAisFile(e.target.files[0] || null); setCorr(null); }} /></label>
        {aisFile && <button onClick={runAis}>VALIDATE AIS</button>}
        {aisInfo && <div className="result"><p>Rows: {aisInfo.rows} · Vessels: {aisInfo.vessels}</p><p>{aisInfo.validation_warnings?.join(' ') || 'AIS validation passed without warnings.'}</p></div>}
        {sar?.centroid_geographic && aisFile && <button onClick={runCorrelation}>CORRELATE VESSELS</button>}
        {corr?.candidates?.length > 0 && <div className="tablewrap"><table><thead><tr><th>MMSI</th><th>Name</th><th>Distance km</th><th>Time min</th><th>Continuity</th><th>Correlation</th></tr></thead><tbody>{corr.candidates.slice(0,20).map(v => <tr key={v.mmsi}><td>{v.mmsi}</td><td>{v.vessel_name || '—'}</td><td>{fmt(v.nearest_distance_km)}</td><td>{fmt(v.time_difference_minutes,1)}</td><td>{fmt(v.ais_continuity,2)}</td><td>{fmt(v.correlation_score,1)}</td></tr>)}</tbody></table></div>}
      </section>

      <section className="wide card"><h3>HUMAN VALIDATION & REPORT</h3><div className="actions"><button>ACCEPT DETECTION</button><button>REJECT DETECTION</button><button>NEEDS REVIEW</button><button onClick={exportReport}>EXPORT ANALYSIS REPORT</button></div><p className="disclaimer">Investigation lead only — correlation does not establish responsibility.</p></section>
    </main>
  </div>;
}
