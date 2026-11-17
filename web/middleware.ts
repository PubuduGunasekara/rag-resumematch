import { NextResponse, type NextRequest } from "next/server";

// Signed-out visitors go to /login. This only checks that the session
// cookie exists, to avoid a page flash; the API checks the signature and
// the role on every request, so a forged cookie gets nothing.
export function middleware(request: NextRequest) {
  if (request.cookies.has("rm_session")) return NextResponse.next();
  const url = request.nextUrl.clone();
  url.pathname = "/login";
  url.search = request.nextUrl.pathname === "/" ? "" : `?next=${encodeURIComponent(request.nextUrl.pathname)}`;
  return NextResponse.redirect(url);
}

export const config = {
  // Everything except the login page, the API proxy and Next.js assets.
  matcher: ["/((?!login|api|_next|favicon.ico).*)"],
};
