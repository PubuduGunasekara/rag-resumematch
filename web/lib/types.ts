// Mirrors resumematch.api.app.report_to_json. All wording shown to the
// user comes from the Python side so both front ends say the same thing.

export type Status = "shown" | "partly" | "not_found";

export interface Requirement {
  text: string;
  met: boolean;
  status: Status;
  importance: "required" | "preferred" | "unspecified";
  credential: boolean;
  soft: boolean;
  resume_quote: string | null;
  posting_sentence: string | null;
  partly: string[];
}

export interface Job {
  rank: number;
  job_id: string;
  title: string;
  company: string | null;
  url: string | null;
  details: { location?: string | null; work_type?: string | null; level?: string | null; remote?: boolean; salary?: string | null };
  tier: "best" | "potential" | "no";
  reason: string;
  met: number;
  checked: number;
  partly: number;
  requirements: Requirement[];
  highlight: [number, number][];
}

export interface Report {
  headline: string;
  summary: string;
  total_postings: number | null;
  best_job_id: string | null;
  jobs: Job[];
  learn_first: { text: string; jobs: number; credential: boolean }[];
  resume_text: string;
  how: {
    sections: string[];
    retrieved: { title: string; company: string | null; matched_sections: string[] }[];
    corrections: { original: string; corrected: string }[];
    rejected: { requirement: string; claimed_as: string; issue: string }[];
    attempts: number;
  };
}
