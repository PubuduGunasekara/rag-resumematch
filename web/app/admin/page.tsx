"use client";

import { useEffect, useState } from "react";
import RoleGate from "@/components/RoleGate";
import { handleSignedOut } from "@/lib/session";

type Overview = {
  accounts: { email: string; name: string; role_label: string }[];
  recent_sign_ins: { email: string; role: string; at: number }[];
  postings: number | null;
  busy: boolean;
};

function AdminOverview() {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/admin/overview")
      .then(async (res) => {
        if (handleSignedOut(res)) return;
        const body = await res.json();
        if (!res.ok) setError(body?.detail?.message || "Couldn't load the overview.");
        else setData(body);
      })
      .catch(() => setError("Couldn't reach the analysis service. Is the API running on port 8000?"));
  }, []);

  return (
    <>
      <p className="tag">Admin</p>
      <h1 className="serif">System overview</h1>
      {error && <div className="error-box" role="alert"><p className="error">{error}</p></div>}
      {data && (
        <>
          <div className="facts">
            <span>{data.postings ? `${data.postings.toLocaleString()} postings indexed` : "Posting count unknown"}</span>
            <span>{data.busy ? "A resume check is running" : "No resume check running"}</span>
          </div>
          <div className="section">
            <p className="tag">Demo accounts</p>
            {data.accounts.map((a) => (
              <div className="learn" key={a.email}><span>{a.name}<br /><span className="small muted">{a.email}</span></span><span>{a.role_label}</span></div>
            ))}
          </div>
          <div className="section">
            <p className="tag">Recent sign-ins (since the API started)</p>
            {data.recent_sign_ins.length === 0 ? <p className="small muted">None yet.</p> : data.recent_sign_ins.map((s, i) => (
              <div className="learn" key={i}><span>{s.email}</span><span className="small muted">{new Date(s.at * 1000).toLocaleString()}</span></div>
            ))}
          </div>
        </>
      )}
    </>
  );
}

export default function AdminPage() {
  return <RoleGate href="/admin"><AdminOverview /></RoleGate>;
}
