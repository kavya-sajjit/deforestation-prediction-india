"use client";
import { useState } from "react";
import { BASE, PATCH_HA, REGION_LABEL, TIER } from "@/lib/data";
import { EVERYWHERE, nearest } from "@/lib/contacts";

const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);

function alertText(doc, s, tierRow) {
  const r = REGION_LABEL[doc.region] || doc.name;
  return [
    `Forest-loss risk alert · ${r} · hotspot #${s.id}`,
    ``,
    `Location: ${s.lat.toFixed(4)} N, ${s.lon.toFixed(4)} E (Google Maps: https://maps.google.com/?q=${s.lat},${s.lon})`,
    `Area: ${s.squares} squares of 640 m, about ${s.forest_ha} ha of forest`,
    `Risk level: ${TIER[s.tier].label} (average ${pct(s.mean_risk)} chance per square of losing at least 2% of its forest)`,
    tierRow?.hit_rate != null
      ? `How reliable: when we tested on areas the model had never seen, ${pct(tierRow.hit_rate)} of squares rated ${TIER[s.tier].label} really lost forest (average square: ${pct(doc.risk.base_rate)}).`
      : ``,
    ``,
    `This is a statistical forecast from 2019 satellite imagery and past clearing, not evidence of any illegal activity. Labels include plantation harvest and shifting cultivation, so a field check is needed before any action.`,
    `Map and method: https://github.com/kavya-sajjit/deforestation-prediction-india`,
  ].filter((l) => l !== null).join("\n");
}

export default function Hotspots({ doc, selected, onSelect }) {
  const [copied, setCopied] = useState(false);
  const spots = doc?.hotspots || [];
  if (!doc?.risk || !spots.length) return null;
  const s = spots.find((h) => h.id === selected);
  const tierRow = s && doc.risk.tiers.find((t) => t.tier === s.tier);

  if (!s) {
    return (
      <div className="hotspots">
        <div className="hs-head"><span className="dot" /> High-risk areas</div>
        <ol>
          {spots.slice(0, 5).map((h) => (
            <li key={h.id}>
              <button onClick={() => onSelect(h.id)}>
                <span className="hs-n">#{h.id}</span>
                <span className={`chip ${h.tier}`}>{TIER[h.tier].label}</span>
                <span className="hs-a">{h.at_risk_ha} ha at risk</span>
              </button>
            </li>
          ))}
        </ol>
        <div className="hs-foot">ranked by forest likely to be lost · click one</div>
      </div>
    );
  }

  const text = alertText(doc, s, tierRow);
  const copy = async () => {
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1800); } catch {}
  };
  const subject = encodeURIComponent(`Forest-loss risk alert: ${REGION_LABEL[doc.region]} hotspot #${s.id}`);

  return (
    <div className="hotspots open">
      <div className="hs-head">
        <button className="back" onClick={() => onSelect(null)} aria-label="Back to list">←</button>
        Hotspot #{s.id} <span className={`chip ${s.tier}`}>{TIER[s.tier].label}</span>
      </div>
      <div className="kv tight">
        <span>Risk per square</span><b className="accent-ochre">{pct(s.mean_risk)}</b>
        <span>Forest here</span><b>{s.forest_ha} ha</b>
        <span>Likely to be lost</span><b className="accent-rust">{s.at_risk_ha} ha</b>
      </div>
      {tierRow?.hit_rate != null && (
        <p className="hs-proof">
          Tested on unseen areas, <b>{pct(tierRow.hit_rate)}</b> of “{TIER[s.tier].label}” squares really lost forest
          {tierRow.ci?.[0] != null && <> ({pct(tierRow.ci[0])}–{pct(tierRow.ci[1])})</>}. An average square: {pct(doc.risk.base_rate)}.
        </p>
      )}
      <div className="hs-sub">Who to tell</div>
      <ul className="contacts">
        {nearest(doc.region, s).map((c) => (
          <li key={c.name}><a href={c.url} target="_blank" rel="noreferrer">{c.name}</a>
            <span>{c.kind} · {c.city}, ~{c.km} km</span></li>
        ))}
        {EVERYWHERE.slice(0, 1).map((c) => (
          <li key={c.name}><a href={c.url} target="_blank" rel="noreferrer">{c.name}</a><span>{c.what}</span></li>
        ))}
      </ul>
      <div className="hs-actions">
        <button className="btn" onClick={copy}>{copied ? "COPIED" : "COPY ALERT"}</button>
        <a className="btn" href={`mailto:?subject=${subject}&body=${encodeURIComponent(text)}`}>EMAIL</a>
        <a className="btn" href={`${BASE}/data/${doc.region}_hotspots.geojson`} download>MAP FILE</a>
      </div>
      <div className="hs-foot">A forecast, not proof. Ask for a field check before anyone acts.</div>
    </div>
  );
}
