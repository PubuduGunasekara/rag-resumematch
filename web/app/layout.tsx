import type { Metadata } from "next";
import { IBM_Plex_Sans, Source_Serif_4 } from "next/font/google";
import Link from "next/link";
import "./globals.css";

// Self-hosted at build time by next/font: no request to Google at runtime.
const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-sans" });
const serif = Source_Serif_4({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-serif" });

export const metadata: Metadata = {
  title: "ResumeMatch",
  description: "Check your resume against 107,528 real job postings, with every claim verified against what your resume says.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body>
        <header className="site-header">
          <Link href="/" className="wordmark">ResumeMatch</Link>
        </header>
        <main>{children}</main>
        <footer className="site-footer">
          Postings from LinkedIn Job Postings 2023 to 2024 by Arsh Koneru, CC BY-SA 4.0. Listings may have closed.
        </footer>
      </body>
    </html>
  );
}
