import { MODEL_LABEL } from "@/lib/data";

const SHOW = ["cnn_rgbn_hist", "gbm", "bcos_rgbn_hist", "bcos_pre_project", "bcos_pre_rgb",
              "bcos_pre_inflate", "bcos_pre_ndvi", "history_lr", "bcos_rgbn", "bcos_rgb"];

export default function Results({ rows }) {
  if (!rows?.length) return <p className="note">Results appear here after the experiments run.</p>;
  const by = Object.fromEntries(rows.map((r) => [r.experiment, r]));
  const list = SHOW.filter((k) => by[k]);
  const max = Math.max(0.6, ...list.map((k) => by[k].ap));
  const chance = rows[0].chance_ap;
  return (
    <div className="results">
      {list.map((k) => {
        const r = by[k];
        const cls = k.startsWith("bcos") ? "bcos" : k === "gbm" || k === "history_lr" ? "base" : "";
        return (
          <div key={k} className={`row ${cls}`}>
            <span>{MODEL_LABEL[k] || k}</span>
            <span className="track">
              <span className="fill" style={{ width: `${(r.ap / max) * 100}%` }} />
              <span className="chance" style={{ left: `${(chance / max) * 100}%` }} title={`chance ${chance.toFixed(2)}`} />
            </span>
            <span className="v">{r.ap.toFixed(2)}</span>
          </div>
        );
      })}
      <div className="ticks" style={{ justifyContent: "flex-start", gap: 18 }}>
        <span>PR-AUC on spatially held-out patches (higher is better)</span>
        <span className="fc">│ chance = {chance.toFixed(2)}</span>
      </div>
    </div>
  );
}
