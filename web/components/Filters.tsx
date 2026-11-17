"use client";

import { useEffect, useState } from "react";

// Mirrors resumematch.retrieval.filters and the form fields both POST
// endpoints accept.
export interface FilterValues {
  remote_only: boolean;
  state: string;          // one state code, or "" for any
  work_types: string[];
  levels: string[];
  min_salary: string;     // per year, digits only
}

export const NO_FILTERS: FilterValues = { remote_only: false, state: "", work_types: [], levels: [], min_salary: "" };

type Option = { value: string; postings: number };
interface Options {
  postings: number; remote: number; with_salary: number; with_level: number;
  states: Option[]; work_types: Option[]; levels: Option[];
}

export function activeCount(f: FilterValues): number {
  return [f.remote_only, !!f.state, f.work_types.length > 0, f.levels.length > 0, !!f.min_salary].filter(Boolean).length;
}

export function appendFilters(body: FormData, f: FilterValues) {
  body.append("remote_only", String(f.remote_only));
  body.append("states", f.state);
  body.append("work_types", f.work_types.join(","));
  body.append("levels", f.levels.join(","));
  body.append("min_salary", f.min_salary);
}

const pct = (n: number, of: number) => (of ? `${Math.round((100 * n) / of)}%` : "");

export default function Filters({ value, onChange, disabled }: {
  value: FilterValues; onChange: (f: FilterValues) => void; disabled?: boolean;
}) {
  const [options, setOptions] = useState<Options | null>(null);
  useEffect(() => {
    fetch("/api/filter-options").then((r) => (r.ok ? r.json() : null)).then(setOptions).catch(() => setOptions(null));
  }, []);

  const toggle = (key: "work_types" | "levels", v: string) =>
    onChange({ ...value, [key]: value[key].includes(v) ? value[key].filter((x) => x !== v) : [...value[key], v] });
  const n = activeCount(value);

  return (
    <details className="filters" open={n > 0}>
      <summary>Filters{n ? ` (${n} on)` : ""}</summary>
      {!options ? <p className="small muted">Loading filter choices</p> : (
        <fieldset disabled={disabled}>
          <label className="check">
            <input type="checkbox" checked={value.remote_only} onChange={(e) => onChange({ ...value, remote_only: e.target.checked })} />
            Remote allowed <span className="muted">({pct(options.remote, options.postings)} of postings)</span>
          </label>

          <label>
            State
            <select value={value.state} onChange={(e) => onChange({ ...value, state: e.target.value })}>
              <option value="">Any state</option>
              {options.states.map((s) => <option key={s.value} value={s.value}>{s.value} ({s.postings.toLocaleString()})</option>)}
            </select>
          </label>

          <div>
            <p className="label">Work type</p>
            <div className="chips">
              {options.work_types.map((w) => (
                <label key={w.value} className="check"><input type="checkbox" checked={value.work_types.includes(w.value)} onChange={() => toggle("work_types", w.value)} />{w.value}</label>
              ))}
            </div>
          </div>

          <div>
            <p className="label">Level <span className="muted">(stated on {pct(options.with_level, options.postings)} of postings)</span></p>
            <div className="chips">
              {options.levels.map((l) => (
                <label key={l.value} className="check"><input type="checkbox" checked={value.levels.includes(l.value)} onChange={() => toggle("levels", l.value)} />{l.value}</label>
              ))}
            </div>
          </div>

          <label>
            Minimum salary per year, USD <span className="muted">(listed on {pct(options.with_salary, options.postings)} of postings)</span>
            <input type="text" inputMode="numeric" placeholder="for example 80000" value={value.min_salary}
                   onChange={(e) => onChange({ ...value, min_salary: e.target.value.replace(/[^\d]/g, "").slice(0, 7) })} />
          </label>

          {n > 0 && <button type="button" className="link-button" onClick={() => onChange(NO_FILTERS)}>Clear filters</button>}
        </fieldset>
      )}
    </details>
  );
}

/** The "Filtered to ..." line shown on a report. */
export function FilterNote({ filters }: { filters: { applied: string; caveats: string[]; postings: number | null } | null }) {
  if (!filters) return null;
  return (
    <p className="filter-note">
      Filtered: {filters.applied}
      {filters.postings !== null ? `. ${filters.postings.toLocaleString()} indexed postings of any title pass these filters.` : "."}
      {filters.caveats.length > 0 && <span className="muted"> {filters.caveats.join(" ")}</span>}
    </p>
  );
}
