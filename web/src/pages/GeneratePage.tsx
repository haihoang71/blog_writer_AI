import { FormEvent, useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";

export default function GeneratePage() {
  const faults = useQuery({ queryKey: ["faults"], queryFn: api.faults });
  const [topic, setTopic] = useState("LangGraph checkpointing for production agents");
  const [hitl, setHitl] = useState(false);
  const [scenario, setScenario] = useState("none");
  const [busy, setBusy] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState("approve");

  const status = useQuery({
    queryKey: ["status", taskId],
    queryFn: () => api.status(taskId!),
    enabled: Boolean(taskId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s && ["completed", "failed", "timeout"].includes(s) ? false : 800;
    },
  });

  useEffect(() => {
    if (!runId) return;
    const source = new EventSource(`/api/v1/runs/${runId}/stream`);
    const onAny = (ev: MessageEvent) => {
      setEvents((prev) => [...prev.slice(-40), `${ev.type}: ${ev.data.slice(0, 180)}`]);
    };
    source.onmessage = onAny;
    ["node_started", "node_completed", "fault_injected", "run_completed", "done", "hitl_paused"].forEach(
      (name) => source.addEventListener(name, onAny as EventListener),
    );
    return () => source.close();
  }, [runId]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setEvents([]);
    try {
      const created = await api.generate({ topic, enable_hitl: hitl, fault_scenario: scenario });
      setTaskId(created.task_id);
      setRunId(created.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onReview() {
    if (!taskId) return;
    await api.review(taskId, feedback);
    await status.refetch();
  }

  const interrupt = status.data?.result?.interrupt;
  const scenarios = faults.data?.scenarios ?? ["none"];
  const mode = status.data?.execution_mode;
  const modeClass = useMemo(() => {
    if (mode === "live") return "live";
    if (mode === "mock_fallback") return "fallback";
    return "mock";
  }, [mode]);

  return (
    <>
      <div className="card">
        <h2>Generate a blog post</h2>
        <p className="muted">
          Clean LangGraph path: input_guard → runtime_probe → planner → researcher →
          writer → critic → human_review → output_guard. Faults are a bounded side
          branch at runtime_probe and do not rewrite agent prompts.
        </p>
        <form onSubmit={onSubmit}>
          <label>Topic</label>
          <input value={topic} onChange={(e) => setTopic(e.target.value)} />
          <div className="row" style={{ marginTop: 12 }}>
            <div style={{ flex: 1 }}>
              <label>Fault scenario</label>
              <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
                {scenarios.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <label className="row" style={{ marginTop: 18 }}>
              <input type="checkbox" checked={hitl} onChange={(e) => setHitl(e.target.checked)} style={{ width: "auto" }} />
              Human review (HITL)
            </label>
            <button type="submit" disabled={busy} style={{ marginTop: 18 }}>
              {busy ? "Queuing…" : "Generate"}
            </button>
          </div>
        </form>
        {error && <p className="badge error">{error}</p>}
      </div>

      {taskId && (
        <div className="card">
          <div className="row">
            <strong>Status:</strong> {status.data?.status ?? "queued"}
            {mode && <span className={`badge ${modeClass}`}>{mode}</span>}
            {runId && (
              <Link to={`/explorer/${runId}`} className="badge">
                Open trace {runId.slice(0, 8)}
              </Link>
            )}
          </div>
          {interrupt ? (
            <div style={{ marginTop: 12 }}>
              <p className="muted">{String(interrupt.message ?? "Review required")}</p>
              <pre>{String(interrupt.draft_preview ?? "")}</pre>
              <label>Feedback</label>
              <input value={feedback} onChange={(e) => setFeedback(e.target.value)} />
              <button type="button" onClick={onReview} style={{ marginTop: 8 }}>
                Resume
              </button>
            </div>
          ) : null}
          {status.data?.result?.final_post ? (
            <pre style={{ marginTop: 12 }}>{String(status.data.result.final_post).slice(0, 4000)}</pre>
          ) : null}
          <h3>Live events</h3>
          <pre>{events.join("\n") || "waiting for SSE…"}</pre>
        </div>
      )}
    </>
  );
}
