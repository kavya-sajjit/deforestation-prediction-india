"use client";
import { useEffect, useMemo, useState } from "react";

// A tree drawn in characters. Year by year, a cleared edge eats into the canopy
// from one side, the way loss spreads from an existing frontier.
const W = 54, H = 46;
const RAMP = " .:-=+*#%@";

function rand(i) { const x = Math.sin(i * 12.9898) * 43758.5453; return x - Math.floor(x); }

function buildTree() {
  const cells = [];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const cx = W / 2, nx = (x - cx) / (W / 2);
      let d = 0, kind = "canopy";
      // three overlapping canopy lobes
      const lobes = [[0, 13, 15, 12], [-9, 20, 11, 9], [9, 20, 11, 9], [0, 24, 17, 8]];
      for (const [ox, oy, rx, ry] of lobes) {
        const v = 1 - (((x - (cx + ox)) / rx) ** 2 + ((y - oy) / ry) ** 2);
        d = Math.max(d, v);
      }
      d += (rand(y * W + x) - .5) * .18;
      // trunk and roots
      if (y > 30 && Math.abs(x - cx) < 2.2 + (y - 30) * 0.08) { d = .75; kind = "trunk"; }
      if (y > 41 && Math.abs(nx) < .55) { d = Math.max(d, .35 - Math.abs(nx) * .4); kind = d > .2 ? "soil" : kind; }
      const ch = d <= 0.12 ? " " : RAMP[Math.min(RAMP.length - 1, Math.floor(d * RAMP.length * 1.15) + 1)];
      // order in which cells are cleared: from the lower right edge inward, with noise
      const order = (x / W) * .65 + (y / H) * .2 + rand(x * 7 + y * 131) * .25;
      cells.push({ ch, kind, order });
    }
  }
  return cells;
}

export default function AsciiTree() {
  const cells = useMemo(buildTree, []);
  const [year, setYear] = useState(2001);

  useEffect(() => {
    const reduce = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { setYear(2023); return; }
    const id = setInterval(() => setYear((y) => (y >= 2027 ? 2001 : y + 1)), 520);
    return () => clearInterval(id);
  }, []);

  const shown = Math.min(year, 2023);
  const cleared = Math.max(0, (shown - 2001) / 22) * 0.55; // share of canopy gone by this year
  const threshold = 1 - cleared;

  const rows = [];
  for (let y = 0; y < H; y++) {
    const spans = [];
    let run = "", runCls = null;
    for (let x = 0; x < W; x++) {
      const c = cells[y * W + x];
      let cls = c.kind === "soil" ? "soil" : c.kind === "trunk" ? "soil" : "";
      if (c.kind === "canopy" && c.ch !== " " && c.order > threshold) cls = "lost";
      if (cls !== runCls && run) { spans.push(<span key={x} className={runCls || undefined}>{run}</span>); run = ""; }
      runCls = cls; run += c.ch;
    }
    if (run) spans.push(<span key="e" className={runCls || undefined}>{run}</span>);
    rows.push(<div key={y}>{spans}</div>);
  }

  return (
    <div className="ascii-wrap" aria-hidden="true">
      <pre className="ascii">{rows}</pre>
      <div className="ascii-year">{year > 2023 ? "2024+  ·  FORECAST?" : `${shown}  ·  ${Math.round(cleared * 100)}% CLEARED`}</div>
    </div>
  );
}
