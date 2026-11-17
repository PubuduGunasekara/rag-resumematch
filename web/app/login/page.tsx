"use client";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { homeFor, type Role } from "@/lib/session";

const DEMO_ACCOUNTS: { email: string; label: string; does: string }[] = [
  { email: "seeker@example.com", label: "Job seeker", does: "Check a resume against postings, explore roles" },
  { email: "recruiter@example.com", label: "Recruiter", does: "Explore what postings for a role ask for" },
  { email: "advisor@example.com", label: "Career advisor", does: "Check students' resumes, explore roles" },
  { email: "admin@example.com", label: "Admin", does: "Everything, plus accounts and sign-ins" },
];

function LoginForm() {
  const next = useSearchParams().get("next");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setWorking(true); setError(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) { setError(body?.detail?.message || "Sign-in failed. Try again."); return; }
      // Only same-site paths are followed after sign-in.
      const target = next && next.startsWith("/") && !next.startsWith("//") ? next : homeFor(body.user.role as Role);
      window.location.href = target;
    } catch {
      setError("Couldn't reach the analysis service. Is the API running on port 8000?");
    } finally {
      setWorking(false);
    }
  }

  return (
    <>
      <p className="tag">Sign in</p>
      <h1 className="serif">Sign in to ResumeMatch</h1>
      <p className="lede">This is a demo with one account per role. All four share the demo password set in the server&apos;s <code>.env</code> file.</p>

      <form className="role-form" onSubmit={submit}>
        <label>
          Email
          <input type="text" inputMode="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>
          Password
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>
        <div className="row">
          <button className="button" type="submit" disabled={working}>{working ? "Signing in" : "Sign in"}</button>
        </div>
      </form>
      {error && <div className="error-box" role="alert"><p className="error">{error}</p></div>}

      <div className="section">
        <p className="tag">Demo accounts</p>
        {DEMO_ACCOUNTS.map((a) => (
          <div className="learn" key={a.email}>
            <span><b style={{ fontWeight: 500 }}>{a.label}</b><br /><span className="small muted">{a.does}</span></span>
            <button type="button" className="button ghost" onClick={() => setEmail(a.email)}>Use {a.email.split("@")[0]}</button>
          </div>
        ))}
      </div>
    </>
  );
}

export default function LoginPage() {
  return <Suspense><LoginForm /></Suspense>;
}
