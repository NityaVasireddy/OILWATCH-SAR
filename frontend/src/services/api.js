import axios from 'axios';

export const api = axios.create({ baseURL: 'http://127.0.0.1:8000/api' });

export async function health() { return (await api.get('/health')).data; }
export async function detect(file) { const f = new FormData(); f.append('file', file); return (await api.post('/detect', f)).data; }
export async function drift(data) { return (await api.post('/drift', data)).data; }
export async function ais(file) { const f = new FormData(); f.append('file', file); return (await api.post('/ais/upload', f)).data; }
export async function correlate(file, data) {
  const f = new FormData();
  f.append('file', file);
  f.append('release_lat', String(data.release_lat));
  f.append('release_lon', String(data.release_lon));
  if (data.release_time) f.append('release_time', data.release_time);
  return (await api.post('/correlate', f)).data;
}
export async function report(payload) { return (await api.post('/report', payload)).data; }
