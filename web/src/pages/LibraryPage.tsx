import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";

export default function LibraryPage() {
  const posts = useQuery({ queryKey: ["posts"], queryFn: api.posts });
  const [active, setActive] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: ["post", active],
    queryFn: () => api.post(active!),
    enabled: Boolean(active),
  });
  return (
    <div className="grid2">
      <div className="card">
        <h2>Post library</h2>
        {(posts.data?.posts ?? []).map((post) => {
          const id = String(post.id ?? post.post_id ?? "");
          return (
            <button key={id} className="secondary" style={{ display: "block", width: "100%", marginBottom: 8 }} onClick={() => setActive(id)}>
              {String(post.title ?? post.topic ?? id)}
            </button>
          );
        })}
        {!(posts.data?.posts ?? []).length && <p className="muted">No posts yet.</p>}
      </div>
      <div className="card">
        <h2>Preview</h2>
        {detail.data ? <pre>{JSON.stringify(detail.data, null, 2).slice(0, 8000)}</pre> : <p className="muted">Pick a post.</p>}
      </div>
    </div>
  );
}
