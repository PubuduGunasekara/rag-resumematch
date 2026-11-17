"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

export type Role = "seeker" | "recruiter" | "advisor" | "admin";
export interface User { email: string; name: string; role: Role; role_label: string }

// Mirrors the role checks in resumematch/api/auth.py and app.py. The API
// enforces them; this list only decides which links to show.
export const PAGES: { href: string; label: string; roles: Role[] }[] = [
  { href: "/", label: "Check a resume", roles: ["seeker", "advisor", "admin"] },
  { href: "/role", label: "Explore a role", roles: ["seeker", "recruiter", "advisor", "admin"] },
  { href: "/recruit", label: "Rank candidates", roles: ["recruiter", "admin"] },
  { href: "/admin", label: "Admin", roles: ["admin"] },
];

export function homeFor(role: Role): string {
  return role === "admin" ? "/admin" : role === "recruiter" ? "/recruit" : "/";
}

export function canOpen(user: User | null, href: string): boolean {
  const page = PAGES.find((p) => p.href === href);
  return !!user && !!page && page.roles.includes(user.role);
}

type Session = { user: User | null; loading: boolean; refresh: () => Promise<void>; signOut: () => Promise<void> };
const SessionContext = createContext<Session>({ user: null, loading: true, refresh: async () => {}, signOut: async () => {} });

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/me");
      setUser(res.ok ? (await res.json()).user : null);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const signOut = useCallback(async () => {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    setUser(null);
    window.location.href = "/login";
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  return <SessionContext.Provider value={{ user, loading, refresh, signOut }}>{children}</SessionContext.Provider>;
}

export const useSession = () => useContext(SessionContext);

/** Send the browser to /login when an API call says the session is gone. */
export function handleSignedOut(res: Response): boolean {
  if (res.status !== 401) return false;
  window.location.href = `/login?next=${encodeURIComponent(window.location.pathname)}`;
  return true;
}
