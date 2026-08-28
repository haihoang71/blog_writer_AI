export type RunRecord = {
  run_id: string;
  task_id: string;
  langfuse_trace_id?: string | null;
  topic?: string;
  fault_scenario?: string;
  execution_mode?: string;
  provider?: string;
  model?: string;
  status: string;
  enable_hitl?: boolean;
  error?: string | null;
  started_at?: string | null;
  ended_at?: string | null;
};

export type Span = {
  id: string;
  parent_span_id?: string | null;
  sequence_index: number;
  agent_name: string;
  name: string;
  span_type?: string;
  started_at?: string;
  ended_at?: string;
  duration_ms?: number | null;
  self_time_ms?: number | null;
  status: string;
  status_message?: string | null;
  error_class?: string | null;
  input?: unknown;
  output?: unknown;
  input_tokens?: number;
  output_tokens?: number;
  cost_usd?: number | null;
  usage_source?: string;
  depth?: number;
  reads_state_keys?: string[] | null;
  writes_state_keys?: string[] | null;
  input_hash?: string | null;
};

export type UsageRollup = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  cost_label: string;
  sources: string[];
  disclaimer: string;
};

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch("/health").then((r) => parse<{ execution_mode: string; openai: boolean; langfuse: boolean; tavily: boolean }>(r)),
  faults: () => fetch("/api/v1/faults").then((r) => parse<{ scenarios: string[] }>(r)),
  generate: (body: { topic: string; enable_hitl: boolean; fault_scenario: string }) =>
    fetch("/api/v1/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => parse<{ task_id: string; run_id: string; status: string; message: string }>(r)),
  status: (taskId: string) =>
    fetch(`/api/v1/status/${taskId}`).then((r) =>
      parse<{
        task_id: string;
        run_id?: string;
        status: string;
        result?: { interrupt?: Record<string, unknown>; final_post?: string; word_count?: number };
        error?: string | null;
        execution_mode?: string;
        fault_scenario?: string;
      }>(r),
    ),
  review: (taskId: string, feedback: string) =>
    fetch("/api/v1/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, feedback }),
    }).then((r) => parse<{ status: string; task_id: string; run_id?: string }>(r)),
  runs: () => fetch("/api/v1/runs").then((r) => parse<{ runs: RunRecord[] }>(r)),
  run: (id: string) => fetch(`/api/v1/runs/${id}`).then((r) => parse<RunRecord>(r)),
  trace: (id: string) =>
    fetch(`/api/v1/runs/${id}/trace`).then((r) =>
      parse<{
        run: RunRecord;
        normalized: { span_count: number; agent_count: number; error_count: number; spans: Span[] };
        usage: UsageRollup;
        graph_hint: string[];
      }>(r),
    ),
  raw: (id: string) => fetch(`/api/v1/runs/${id}/trace/raw`).then((r) => parse<unknown>(r)),
  usage: (id: string) => fetch(`/api/v1/runs/${id}/usage`).then((r) => parse<UsageRollup>(r)),
  posts: () => fetch("/api/v1/posts").then((r) => parse<{ posts: Array<Record<string, unknown>> }>(r)),
  post: (id: string) => fetch(`/api/v1/posts/${id}`).then((r) => parse<Record<string, unknown>>(r)),
  sandbox: (code: string) =>
    fetch("/api/v1/sandbox/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    }).then((r) => parse<Record<string, unknown>>(r)),
};
