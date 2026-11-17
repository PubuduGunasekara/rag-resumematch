"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { PAGES, useSession } from "@/lib/session";

export default function Nav() {
  const path = usePathname();
  const { user, signOut } = useSession();
  if (!user || path === "/login") return null;
  return (
    <nav className="site-nav" aria-label="Main">
      {PAGES.filter((p) => p.roles.includes(user.role)).map((p) => (
        <Link key={p.href} href={p.href} aria-current={path === p.href ? "page" : undefined}>{p.label}</Link>
      ))}
      <span className="who" title={user.email}>{user.name}, {user.role_label}</span>
      <button type="button" className="link-button" onClick={signOut}>Sign out</button>
    </nav>
  );
}
