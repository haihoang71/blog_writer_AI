import { NavLink, Route, Routes } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import GeneratePage from "./pages/GeneratePage";
import ExplorerPage from "./pages/ExplorerPage";
import LibraryPage from "./pages/LibraryPage";
import SandboxPage from "./pages/SandboxPage";

function ModeBadge() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  const mode = health.data?.execution_mode ?? "…";
  const cls = mode === "live" ? "live" : mode.includes("fallback") ? "fallback" : "mock";
  return (
    <span className={`badge ${cls}`} title="LLM execution mode">
      {mode}
      {health.data?.langfuse ? " · langfuse" : " · local traces"}
    </span>
  );
}

export default function App() {
  return (
    <>
      <header className="app">
        <h1>Blog Writer</h1>
        <ModeBadge />
        <nav className="app">
          <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
            Generate
          </NavLink>
          <NavLink to="/explorer" className={({ isActive }) => (isActive ? "active" : "")}>
            Trace Explorer
          </NavLink>
          <NavLink to="/library" className={({ isActive }) => (isActive ? "active" : "")}>
            Library
          </NavLink>
          <NavLink to="/sandbox" className={({ isActive }) => (isActive ? "active" : "")}>
            Sandbox
          </NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<GeneratePage />} />
          <Route path="/explorer" element={<ExplorerPage />} />
          <Route path="/explorer/:runId" element={<ExplorerPage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/sandbox" element={<SandboxPage />} />
        </Routes>
      </main>
    </>
  );
}
