/** API client — thin wrapper around fetch */
const BASE = "";

async function _req(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(BASE + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.error || res.statusText);
  }
  return res.json();
}

export const API = {
  status:           ()                    => _req("GET",  "/api/status"),
  graphs:           (split)               => _req("GET",  `/api/graphs${split ? `?split=${split}` : ""}`),
  graphData:        (id)                  => _req("GET",  `/api/graphs/${encodeURIComponent(id)}`),
  patients:         ()                    => _req("GET",  "/api/patients"),
  inference:        (graphId, debug)      => _req("POST", "/api/inference",         { graph_id:   graphId,    debug }),
  inferencePatient: (patientId, debug)    => _req("POST", "/api/inference_patient",  { patient_id: patientId,  debug }),
  stats:            ()                    => _req("GET",  "/api/stats"),
  reload:           ()                    => _req("POST", "/api/reload"),
  checkpoints:      ()                    => _req("GET",  "/api/checkpoints"),
  selectModel:      (name)               => _req("POST", "/api/select_model", { name }),
};
