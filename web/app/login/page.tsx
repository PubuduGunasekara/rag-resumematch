"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { homeFor, type Role } from "@/lib/session";

type DemoAccount = { email: string; role_label: string; does: string };

function LoginForm() {
  const next = useSearchParams().get("next");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const [accounts, setAccounts] = useState<DemoAccount[]>([]);
  const [demoPassword, setDemoPassword] = useState<string | null>(null);
  const submitButton = useRef<HTMLButtonElement>(null);

  // The API lists the demo accounts, and the shared password when the
  // operator has made it public (DEMO_PASSWORD_PUBLIC=1).
  useEffect(() => {
    fetch("/api/auth/demo-accounts")
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => { if (b) { setAccounts(b.accounts); setDemoPassword(b.password); } })
      .catch(() => {});
  }, []);

  function pick(account: DemoAccount) {
    setEmail(account.email);
    if (demoPassword) setPassword(demoPassword);
    setError(null);
    submitButton.current?.focus();
  }

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
      <p className="lede">
        This is a demo with one account per role.{" "}
        {demoPassword
          ? "Pick a role below to fill in its email and password, then sign in."
          : "All four share the demo password set in the server's .env file."}
      </p>

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
          <button ref={submitButton} className="button" type="submit" disabled={working}>{working ? "Signing in" : "Sign in"}</button>
        </div>
      </form>
      {error && <div className="error-box" role="alert"><p className="error">{error}</p></div>}

      {accounts.length > 0 && (
        <div className="section">
          <p className="tag">Demo accounts</p>
          {accounts.map((a) => (
            <div className="learn" key={a.email}>
              <span><b style={{ fontWeight: 500 }}>{a.role_label}</b><br /><span className="small muted">{a.does}</span></span>
              <button type="button" className="button ghost" aria-pressed={email === a.email} onClick={() => pick(a)}
                      aria-label={`Fill in the ${a.role_label} account`}>
                {demoPassword ? "Fill in" : "Use email"}
              </button>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export default function LoginPage() {
  return <Suspense><LoginForm /></Suspense>;
}
