import { useState } from "react";
import { api } from "../api";

const SAMPLE = `print(sum(range(10)))
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("demo")
`;

export default function SandboxPage() {
  const [code, setCode] = useState(SAMPLE);
  const [out, setOut] = useState<string>("");
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      const result = await api.sandbox(code);
      setOut(JSON.stringify(result, null, 2));
    } catch (err) {
      setOut(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Local sandbox</h2>
      <p className="muted">Same subprocess sandbox the Critic uses. No remote code execution.</p>
      <textarea value={code} onChange={(e) => setCode(e.target.value)} style={{ minHeight: 180 }} />
      <button type="button" onClick={run} disabled={busy} style={{ marginTop: 10 }}>
        {busy ? "Running…" : "Execute"}
      </button>
      <pre>{out}</pre>
    </div>
  );
}
