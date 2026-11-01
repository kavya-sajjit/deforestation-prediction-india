export const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";

export const REGION_ORDER = ["western_ghats", "odisha_mining", "northeast_hills", "central_india"];

export const REGION_LABEL = {
  western_ghats: "Western Ghats",
  odisha_mining: "Odisha mining belt",
  northeast_hills: "Northeast hills",
  central_india: "Central India",
};

export const MODEL_LABEL = {
  bcos_rgbn_hist: "B-cos + history (interpretable)",
  cnn_rgbn_hist: "Black-box CNN + history",
  gbm: "Gradient boosting",
  history_lr: "Past-loss history only",
  bcos_pre_rgb: "B-cos, pretrained, colour only",
  bcos_pre_project: "B-cos, pretrained, NIR via projection",
  bcos_pre_inflate: "B-cos, pretrained, NIR via inflation",
  bcos_pre_ndvi: "B-cos, pretrained, NDVI fusion",
  cnn_rgbn: "Black-box CNN, imagery only",
  bcos_rgbn: "B-cos from scratch, RGB+NIR",
  bcos_rgb: "B-cos from scratch, RGB",
};

export const PATCH_HA = 40.96; // 640 m x 640 m

export async function loadRegion(region) {
  const res = await fetch(`${BASE}/data/${region}.json`);
  if (!res.ok) throw new Error(`could not load ${region}`);
  return res.json();
}

/** Cumulative share of the patch lost up to and including `year`. */
export function lostBy(p, year) {
  if (!p.l) return 0;
  let s = 0;
  for (const [y, v] of Object.entries(p.l)) if (+y <= year) s += v;
  return s;
}

// Risk tiers, cut on CALIBRATED risk (see deforest/risk_levels.py). Keep in sync with config.RISK_TIERS.
export const TIER = {
  high: { label: "High", min: 0.5, color: "#c8643b" },
  elevated: { label: "Elevated", min: 0.25, color: "#d9a441" },
  low: { label: "Low", min: 0, color: "#26372b" },
};
