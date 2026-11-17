"use client";

import Link from "next/link";
import { canOpen, homeFor, useSession } from "@/lib/session";

/** Renders the page only for roles allowed to open it; others get a plain
 * message and a link to their own start page. The API enforces the same rule. */
export default function RoleGate({ href, children }: { href: string; children: React.ReactNode }) {
  const { user, loading } = useSession();
  if (loading) return <p className="small muted">Loading</p>;
  if (!user) return <p className="small muted">Sign in to continue. <Link href="/login">Sign in</Link></p>;
  if (!canOpen(user, href)) {
    return (
      <div className="error-box" role="alert">
        <p className="error">This page isn&apos;t available to the {user.role_label} role.</p>
        <p className="small"><Link href={homeFor(user.role)}>Go to your start page</Link></p>
      </div>
    );
  }
  return <>{children}</>;
}
