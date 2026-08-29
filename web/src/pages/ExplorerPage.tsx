import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api, type Span } from "../api";

function statusClass(status: string) {
  if (status === "error" || status === "timeout") return "error";
  if (status === "ok") return "live";
  return "mock";
}

export default function ExplorerPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState<"overview" | "graph" | "timeline" | "span" | "raw">("overview");
  const [selected, setSelected] = useState<Span | null>(null);

  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs, refetchInterval: 4000 });
  const trace = useQuery({
    queryKey: ["trace", runId],
    queryFn: () => api.trace(runId!),
    enabled: Boolean(runId),
    refetchInterval: 2000,
  });
  const raw = useQuery({
    queryKey: ["raw", runId],
    queryFn: () => api.raw(runId!),
    enabled: Boolean(runId) && tab === "raw",
  });

  const spans = trace.data?.normalized.spans ?? [];
  const timelineStats = useMemo(() => {
    const totals = new Map<string, { agent: string; total: number; calls: number }>();
    spans.forEach((span) => {
      const duration = Math.max(0, Number(span.duration_ms ?? 0));
      const current = totals.get(span.agent_name) ?? { agent: span.agent_name, total: 0, calls: 0 };
      current.total += duration;
      current.calls += 1;
      totals.set(span.agent_name, current);
    });
    const byAgent = Array.from(totals.values());
    const maxTotal = Math.max(...byAgent.map((item) => item.total), 1);
    const maxCall = Math.max(...spans.map((span) => Number(span.duration_ms ?? 0)), 1);
    return { byAgent, maxTotal, maxCall };
  }, [spans]);
  const { nodes, edges } = useMemo(() => {
    const ns: Node[] = spans.map((span, index) => ({
      id: span.id,
      data: { label: `${span.agent_name}\n${span.status}` },
      position: { x: (span.depth ?? 0) * 220, y: index * 70 },
      style: {
        border: span.status === "error" || span.status === "timeout" ? "1px solid #ff7a7a" : "1px solid #2a3040",
        background: "#1d2230",
        color: "#e6e8ee",
        fontSize: 11,
        width: 180,
      },
    }));
    const es: Edge[] = spans
      .filter((span) => span.parent_span_id)
      .map((span) => ({
        id: `${span.parent_span_id}-${span.id}`,
        source: span.parent_span_id as string,
        target: span.id,
      }));
    return { nodes: ns, edges: es };
  }, [spans]);

  return (
    <>
      <div className="card">
        <h2>Trace Explorer</h2>
        <p className="muted">
          Observed graph from the Langfuse trace (SQLite is a temporary
          fallback while the remote trace is pending; this is not the designed
          LangGraph). Usage
          numbers are labeled Provider reported / Langfuse estimated / Synthetic /
          Unavailable — synthetic is never a bill.
        </p>
        <label>Run</label>
        <select
          value={runId ?? ""}
          onChange={(e) => navigate(e.target.value ? `/explorer/${e.target.value}` : "/explorer")}
        >
          <option value="">Select a run…</option>
          {(runs.data?.runs ?? []).map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {run.status} · {run.fault_scenario} · {run.topic?.slice(0, 48)}
            </option>
          ))}
        </select>
      </div>

      {trace.data && (
        <>
          <div className="card">
            <div className="row">
              <span className={`badge ${statusClass(trace.data.run.status)}`}>{trace.data.run.status}</span>
              <span className={`badge ${trace.data.run.execution_mode === "live" ? "live" : "mock"}`}>
                {trace.data.run.execution_mode}
              </span>
              <span className={`badge ${trace.data.source === "langfuse" ? "live" : "mock"}`}>
                trace: {trace.data.source === "langfuse" ? "Langfuse thật" : "SQLite fallback"}
              </span>
              <span className="badge">fault: {trace.data.run.fault_scenario}</span>
              <span className="badge">{trace.data.usage.cost_label}</span>
              <span className="muted">
                {trace.data.normalized.span_count} spans · {trace.data.normalized.agent_count} agents ·{" "}
                {trace.data.normalized.error_count} errors
              </span>
            </div>
            <p>
              tokens in/out {trace.data.usage.input_tokens}/{trace.data.usage.output_tokens} · cost{" "}
              {trace.data.usage.cost_usd === null ? "unavailable" : `$${trace.data.usage.cost_usd.toFixed(4)}`}
            </p>
            {trace.data.langfuse_pending && (
              <p className="muted">Langfuse chưa trả observations; đang hiển thị bản ghi SQLite tạm thời. Làm mới sau vài giây.</p>
            )}
            <p className="disclaimer">{trace.data.usage.disclaimer}</p>
            <p className="muted">
              Langfuse trace ID: <code>{trace.data.run.langfuse_trace_id ?? "chưa có"}</code>
              {trace.data.remote_trace?.name ? ` · ${String(trace.data.remote_trace.name)}` : ""}
            </p>
            <div className="tabs">
              {(["overview", "graph", "timeline", "span", "raw"] as const).map((name) => (
                <button key={name} className={tab === name ? "on" : ""} onClick={() => setTab(name)}>
                  {name}
                </button>
              ))}
            </div>
            {tab === "overview" && (
              <div>
                <p className="muted">Designed path (for orientation): {trace.data.graph_hint.join(" → ")}</p>
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>agent</th>
                      <th>status</th>
                      <th>ms</th>
                      <th>tokens</th>
                      <th>cost</th>
                      <th>usage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {spans.map((span) => (
                      <tr key={span.id} onClick={() => { setSelected(span); setTab("span"); }}>
                        <td>{span.sequence_index}</td>
                        <td>{span.agent_name}</td>
                        <td>{span.status}</td>
                        <td>{span.duration_ms ?? "—"}</td>
                        <td>
                          {span.input_tokens}/{span.output_tokens}
                        </td>
                        <td>{span.cost_usd == null ? "—" : `$${span.cost_usd.toFixed(4)}`}</td>
                        <td>{span.usage_source}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {tab === "graph" && (
              <div className="flow">
                <ReactFlow nodes={nodes} edges={edges} fitView onNodeClick={(_e, node) => {
                  const span = spans.find((item) => item.id === node.id);
                  if (span) { setSelected(span); setTab("span"); }
                }}>
                  <Background />
                  <Controls />
                  <MiniMap />
                </ReactFlow>
              </div>
            )}
            {tab === "timeline" && (
              <div>
                <h3>Time by node (all calls aggregated)</h3>
                <p className="muted">
                  Total wall time per agent. Revisions are included, so five Writer calls are summed together.
                </p>
                {timelineStats.byAgent.map((item) => (
                  <div key={item.agent} className="timeline-item timeline-total">
                    <div className="timeline-label">{item.agent}</div>
                    <div className="timeline-track">
                      <div className="bar" style={{ width: `${(item.total / timelineStats.maxTotal) * 100}%` }} />
                    </div>
                    <div className="muted timeline-value">{Math.round(item.total)} ms · {item.calls} call{item.calls === 1 ? "" : "s"}</div>
                  </div>
                ))}

                <h3 style={{ marginTop: 24 }}>Time by call (chronological)</h3>
                <p className="muted">
                  Each observation is shown separately in trace order. Click a row to inspect its input/output and tokens.
                </p>
                {spans.map((span, index) => {
                  const duration = Math.max(0, Number(span.duration_ms ?? 0));
                  return (
                    <div key={span.id} className="timeline-item" onClick={() => { setSelected(span); setTab("span"); }}>
                      <div className="timeline-label">{index + 1}. {span.agent_name}</div>
                      <div className="timeline-track">
                        <div className="bar" style={{ width: `${(duration / timelineStats.maxCall) * 100}%` }} />
                      </div>
                      <div className="muted timeline-value">{Math.round(duration)} ms · {span.status}</div>
                    </div>
                  );
                })}
              </div>
            )}
            {tab === "span" && (
              <div>
                {!selected && <p className="muted">Select a span from overview, graph, or timeline.</p>}
                {selected && (
                  <>
                    <h3>{selected.agent_name} · {selected.name}</h3>
                    <p className="muted">
                      {selected.status} · {selected.span_type ?? "span"} · {selected.usage_source} · hash {selected.input_hash?.slice(0, 12)}
                    </p>
                    <div className="row detail-meta">
                      <span>parent: {selected.parent_span_id ?? "root"}</span>
                      <span>duration: {selected.duration_ms ?? "—"} ms</span>
                      <span>self: {selected.self_time_ms ?? "—"} ms</span>
                      <span>model: {selected.model ?? "—"}</span>
                      <span>tokens: {selected.input_tokens ?? 0} in / {selected.output_tokens ?? 0} out</span>
                      <span>cost: {selected.cost_usd == null ? "unavailable" : `$${selected.cost_usd.toFixed(4)}`}</span>
                    </div>
                    {selected.status_message && <p className="error-text">{selected.status_message}</p>}
                    <div className="grid2">
                      <div>
                        <label>Input</label>
                        <pre>{JSON.stringify(selected.input, null, 2)}</pre>
                      </div>
                      <div>
                        <label>Output</label>
                        <pre>{JSON.stringify(selected.output, null, 2)}</pre>
                      </div>
                    </div>
                    <label>Metadata / state links</label>
                    <pre>{JSON.stringify({
                      ...(selected.metadata ?? {}),
                      reads_state_keys: selected.reads_state_keys,
                      writes_state_keys: selected.writes_state_keys,
                    }, null, 2)}</pre>
                  </>
                )}
              </div>
            )}
            {tab === "raw" && (
              raw.isLoading ? <p className="muted">Loading raw Langfuse payload…</p> :
              raw.error ? <p className="error-text">{String(raw.error)}</p> :
              <pre>{JSON.stringify(raw.data, null, 2)}</pre>
            )}
          </div>
        </>
      )}
    </>
  );
}
