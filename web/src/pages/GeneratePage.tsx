import { Fragment, FormEvent, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";

const GENERATE_STORAGE_KEY = "blog-writer.generate-state.v1";

type SavedResult = {
  interrupt?: Record<string, unknown>;
  final_post?: string;
  word_count?: number;
};

type SavedGeneration = {
  topic: string;
  hitl: boolean;
  scenario: string;
  taskId: string | null;
  runId: string | null;
  events: string[];
  error: string | null;
  feedback: string;
  result?: SavedResult;
};

function renderInline(text: string): ReactNode[] {
  const pattern = /(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\[[^\]]+\]\([^\)]+\)|\*[^*]+\*|_[^_]+_)/g;
  const parts = text.split(pattern);

  return parts.map((part, index) => {
    if (part.startsWith("**") || part.startsWith("__")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("[") && part.includes("](")) {
      const match = part.match(/^\[([^\]]+)\]\(([^\)]+)\)$/);
      if (match) {
        return (
          <a key={index} href={match[2]} target="_blank" rel="noreferrer">
            {match[1]}
          </a>
        );
      }
    }
    if ((part.startsWith("*") && part.endsWith("*")) || (part.startsWith("_") && part.endsWith("_"))) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function MarkdownPreview({ source }: { source: string }) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let paragraph: string[] = [];
  let unordered: string[] = [];
  let ordered: string[] = [];
  let code: string[] | null = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(<p key={`p-${blocks.length}`}>{renderInline(paragraph.join(" "))}</p>);
      paragraph = [];
    }
  };
  const flushLists = () => {
    if (unordered.length) {
      blocks.push(
        <ul key={`ul-${blocks.length}`}>
          {unordered.map((item, index) => <li key={index}>{renderInline(item)}</li>)}
        </ul>,
      );
      unordered = [];
    }
    if (ordered.length) {
      blocks.push(
        <ol key={`ol-${blocks.length}`}>
          {ordered.map((item, index) => <li key={index}>{renderInline(item)}</li>)}
        </ol>,
      );
      ordered = [];
    }
  };

  lines.forEach((line) => {
    if (line.trim().startsWith("```")) {
      flushParagraph();
      flushLists();
      if (code) {
        blocks.push(<pre key={`code-${blocks.length}`}><code>{code.join("\n")}</code></pre>);
        code = null;
      } else {
        code = [];
      }
      return;
    }
    if (code) {
      code.push(line);
      return;
    }
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushLists();
      const level = heading[1].length;
      const Heading = `h${level}` as keyof JSX.IntrinsicElements;
      blocks.push(<Heading key={`h-${blocks.length}`}>{renderInline(heading[2])}</Heading>);
      return;
    }
    const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      if (ordered.length) flushLists();
      unordered.push(bullet[1]);
      return;
    }
    const number = line.match(/^\s*\d+\.\s+(.+)$/);
    if (number) {
      flushParagraph();
      if (unordered.length) flushLists();
      ordered.push(number[1]);
      return;
    }
    if (!line.trim()) {
      flushParagraph();
      flushLists();
      return;
    }
    flushLists();
    paragraph.push(line.trim());
  });

  // The mutation happens inside forEach, which TypeScript cannot model for
  // control-flow narrowing; preserve the declared union explicitly.
  const remainingCode = code as string[] | null;
  if (remainingCode) {
    blocks.push(<pre key={`code-${blocks.length}`}><code>{remainingCode.join("\n")}</code></pre>);
  }
  flushParagraph();
  flushLists();
  return <div className="markdown-preview">{blocks}</div>;
}

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
  const [savedResult, setSavedResult] = useState<SavedResult | undefined>();
  const [showRaw, setShowRaw] = useState(false);
  const [restored, setRestored] = useState(false);

  // Keep the active run in sessionStorage so navigating to Explorer/Library
  // does not discard the generation view when this route unmounts.
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(GENERATE_STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw) as Partial<SavedGeneration>;
        if (typeof saved.topic === "string") setTopic(saved.topic);
        if (typeof saved.hitl === "boolean") setHitl(saved.hitl);
        if (typeof saved.scenario === "string") setScenario(saved.scenario);
        if (typeof saved.taskId === "string" || saved.taskId === null) setTaskId(saved.taskId ?? null);
        if (typeof saved.runId === "string" || saved.runId === null) setRunId(saved.runId ?? null);
        if (Array.isArray(saved.events)) setEvents(saved.events.filter((item): item is string => typeof item === "string"));
        if (typeof saved.error === "string" || saved.error === null) setError(saved.error ?? null);
        if (typeof saved.feedback === "string") setFeedback(saved.feedback);
        if (saved.result) setSavedResult(saved.result);
      }
    } catch {
      // Ignore malformed browser storage and start a fresh generation view.
    } finally {
      setRestored(true);
    }
  }, []);

  const status = useQuery({
    queryKey: ["status", taskId],
    queryFn: () => api.status(taskId!),
    enabled: Boolean(taskId),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s && ["completed", "failed", "timeout", "paused"].includes(s) ? false : 800;
    },
  });

  useEffect(() => {
    if (!restored) return;
    const saved: SavedGeneration = {
      topic, hitl, scenario, taskId, runId, events, error, feedback,
      result: status.data?.result ?? savedResult,
    };
    sessionStorage.setItem(GENERATE_STORAGE_KEY, JSON.stringify(saved));
  }, [restored, topic, hitl, scenario, taskId, runId, events, error, feedback, savedResult, status.data?.result]);

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

  useEffect(() => {
    if (status.data?.result) setSavedResult(status.data.result);
  }, [status.data?.result]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setEvents([]);
    setSavedResult(undefined);
    setShowRaw(false);
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
          {(status.data?.result?.final_post || savedResult?.final_post) ? (
            <div style={{ marginTop: 12 }}>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3 style={{ margin: 0 }}>Generated post</h3>
                <button type="button" className="secondary" onClick={() => setShowRaw((value) => !value)}>
                  {showRaw ? "Preview Markdown" : "View raw Markdown"}
                </button>
              </div>
              {showRaw ? (
                <pre>{String(status.data?.result?.final_post ?? savedResult?.final_post ?? "")}</pre>
              ) : (
                <MarkdownPreview source={String(status.data?.result?.final_post ?? savedResult?.final_post ?? "")} />
              )}
            </div>
          ) : null}
          <h3>Live events</h3>
          <pre>{events.join("\n") || "waiting for SSE…"}</pre>
        </div>
      )}
    </>
  );
}
