"use client";

import Compare from "@/components/Compare";
import { NeedsReport } from "@/components/ReportParts";
import RoleGate from "@/components/RoleGate";
import { useWorkspace } from "@/lib/workspace";

function CompareView() {
  const { report } = useWorkspace();
  if (!report) return <NeedsReport what="Comparing versions" />;
  return (
    <div className="narrow">
      <section className="hero compact">
        <p className="eyebrow">Compare versions</p>
        <h1>Did your edit help?</h1>
        <p className="hero-sub">
          Upload your revised resume. It is checked against the same {report.jobs.length} postings and requirements as your
          report, so you see exactly which requirements your edit changed.
        </p>
      </section>
      <div className="card"><Compare report={report} /></div>
    </div>
  );
}

export default function ComparePage() {
  return <RoleGate href="/compare"><CompareView /></RoleGate>;
}
