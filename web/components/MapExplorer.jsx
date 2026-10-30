"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { loadRegion, lostBy, PATCH_HA, REGION_LABEL, REGION_ORDER } from "@/lib/data";

const FIRST = 2001, LAST = 2023, FORECAST = 2024; // slider value 2024 = model forecast view
const YEARS = Array.from({ length: LAST - FIRST + 1 }, (_, i) => FIRST + i);

const BASEMAP = {
  version: 8,
  sources: {
    sat: {
      type: "raster", tileSize: 256,
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      attribution: "Imagery © Esri, Maxar, Earthstar Geographics · Loss: Hansen/UMD/Google/USGS/NASA",
    },
  },
  layers: [
    { id: "bg", type: "background", paint: { "background-color": "#0d130f" } },
    { id: "sat", type: "raster", source: "sat",
      paint: { "raster-saturation": -0.55, "raster-brightness-max": 0.62, "raster-contrast": 0.05 } },
  ],
};

function toGeoJSON(doc, year) {
  const features = doc.patches.map((p, i) => {
    const { r, c } = p, lon = doc.lon, lat = doc.lat;
    const ring = [[lon[r][c], lat[r][c]], [lon[r][c + 1], lat[r][c + 1]], [lon[r + 1][c + 1], lat[r + 1][c + 1]],
                  [lon[r + 1][c], lat[r + 1][c]], [lon[r][c], lat[r][c]]];
    const forecast = year === FORECAST;
    const lost = lostBy(p, forecast ? 2019 : year);
    const props = {
      i, f: p.f, lost,
      v: forecast ? (p.p ?? -1) : Math.min(1, lost / Math.max(p.f, 1e-3)),
      happened: p.l ? ["2020", "2021", "2022", "2023"].reduce((s, y) => s + (p.l[y] || 0), 0) : 0,
      y: p.y ?? -1,
    };
    return { type: "Feature", id: i, properties: props, geometry: { type: "Polygon", coordinates: [ring] } };
  });
  return { type: "FeatureCollection", features };
}

const OBSERVED_COLOR = ["interpolate", ["linear"], ["get", "v"], 0, "#3f6340", 0.03, "#7a6a3a", 0.12, "#b9673a", 0.35, "#d0582c"];
const RISK_COLOR = ["case", ["<", ["get", "v"], 0], "#26302a",
  ["interpolate", ["linear"], ["get", "v"], 0, "#26372b", 0.35, "#8a7a35", 0.6, "#d9a441", 0.85, "#c8643b"]];

export default function MapExplorer() {
  const box = useRef(null), map = useRef(null);
  const [region, setRegion] = useState(REGION_ORDER[0]);
  const [doc, setDoc] = useState(null);
  const [year, setYear] = useState(LAST);
  const [playing, setPlaying] = useState(false);
  const [showActual, setShowActual] = useState(true);
  const [hover, setHover] = useState(null);
  const [err, setErr] = useState(null);
  const [ready, setReady] = useState(false);

  // load region data
  useEffect(() => {
    let alive = true;
    setErr(null);
    loadRegion(region).then((d) => { if (alive) setDoc(d); })
      .catch((e) => alive && setErr(e.message));
    return () => { alive = false; };
  }, [region]);

  // create map once
  useEffect(() => {
    let m;
    (async () => {
      const maplibregl = (await import("maplibre-gl")).default;
      m = new maplibregl.Map({ container: box.current, style: BASEMAP, center: [80, 22], zoom: 4,
                               attributionControl: { compact: true }, dragRotate: false });
      m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      m.on("load", () => {
        m.addSource("patches", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
        m.addLayer({ id: "fill", type: "fill", source: "patches",
          paint: { "fill-color": OBSERVED_COLOR, "fill-opacity": ["+", 0.25, ["*", 0.5, ["get", "f"]]] } });
        m.addLayer({ id: "actual", type: "line", source: "patches", filter: [">", ["get", "happened"], 0.02],
          paint: { "line-color": "#ebe5d3", "line-width": 1.1, "line-opacity": 0 } });
        m.addLayer({ id: "hover", type: "line", source: "patches", filter: ["==", ["id"], -1],
          paint: { "line-color": "#ebe5d3", "line-width": 2 } });
        map.current = m;
        setReady(true);
      });
      m.on("mousemove", "fill", (e) => {
        const f = e.features?.[0];
        if (!f) return;
        m.setFilter("hover", ["==", ["id"], f.id]);
        setHover({ x: e.point.x, y: e.point.y, i: f.properties.i });
      });
      m.on("mouseleave", "fill", () => { m.setFilter("hover", ["==", ["id"], -1]); setHover(null); });
    })();
    return () => { m && m.remove(); map.current = null; };
  }, []);

  // push data + style whenever region, year or toggle changes
  useEffect(() => {
    const m = map.current;
    if (!m || !doc || !m.getSource("patches")) return;
    m.getSource("patches").setData(toGeoJSON(doc, year));
    const fc = year === FORECAST;
    m.setPaintProperty("fill", "fill-color", fc ? RISK_COLOR : OBSERVED_COLOR);
    m.setPaintProperty("actual", "line-opacity", fc && showActual ? 0.9 : 0);
  }, [doc, year, showActual, ready]);

  // fit to region when it changes
  useEffect(() => {
    const m = map.current;
    if (!m || !doc) return;
    const [x0, y0, x1, y1] = doc.bbox;
    m.fitBounds([[x0, y0], [x1, y1]], { padding: 24, duration: 900 });
  }, [doc?.region, ready]);

  // play
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setYear((y) => (y >= FORECAST ? FIRST : y + 1)), 650);
    return () => clearInterval(id);
  }, [playing]);

  const stats = useMemo(() => {
    if (!doc) return null;
    const forest = doc.patches.reduce((s, p) => s + p.f, 0) * PATCH_HA;
    const perYear = YEARS.map((y) => doc.patches.reduce((s, p) => s + ((p.l && p.l[y]) || 0), 0) * PATCH_HA);
    const shownYear = Math.min(year, LAST);
    const lost = perYear.slice(0, shownYear - FIRST + 1).reduce((a, b) => a + b, 0);
    // forecast check: how much 2020-23 loss falls in the 10% of scored patches the model ranks riskiest
    const scored = doc.patches.filter((p) => p.p !== undefined);
    let capture = null, flagged = 0;
    if (scored.length) {
      const loss = (p) => ["2020", "2021", "2022", "2023"].reduce((s, y) => s + ((p.l && p.l[y]) || 0), 0);
      const sorted = [...scored].sort((a, b) => b.p - a.p);
      flagged = Math.max(1, Math.round(sorted.length * 0.1));
      const total = scored.reduce((s, p) => s + loss(p), 0);
      capture = total > 0 ? sorted.slice(0, flagged).reduce((s, p) => s + loss(p), 0) / total : null;
    }
    return { forest, perYear, lost, capture, flagged, scored: scored.length };
  }, [doc, year]);

  const hp = hover && doc ? doc.patches[hover.i] : null;
  const fc = year === FORECAST;
  const maxBar = stats ? Math.max(...stats.perYear, 1) : 1;

  return (
    <div>
      <div className="tabs" role="tablist" aria-label="Region">
        {REGION_ORDER.map((r) => (
          <button key={r} className="tab" role="tab" aria-selected={r === region} onClick={() => setRegion(r)}>
            {REGION_LABEL[r]}
          </button>
        ))}
      </div>

      <div className="explorer">
        <div className="map" ref={box}>
          {hp && (
            <div className="tooltip" style={{ left: Math.min(hover.x + 14, 9999), top: hover.y + 14 }}>
              <div>forest in 2000 · {(hp.f * 100).toFixed(0)}%</div>
              <div>lost by {fc ? 2019 : year} · {(lostBy(hp, fc ? 2019 : year) * 100).toFixed(1)}%</div>
              {hp.p !== undefined && <div className="accent-ochre">model risk 2020–23 · {(hp.p * 100).toFixed(0)}%</div>}
              {hp.l && <div>actually lost 2020–23 · {(["2020","2021","2022","2023"].reduce((s,y)=>s+((hp.l[y])||0),0)*100).toFixed(1)}%</div>}
            </div>
          )}
        </div>

        <aside className="panel">
          <div>
            <div className="eyebrow">{doc ? doc.biome : "Loading…"}</div>
          </div>
          {err && <div className="note">Could not load map data: {err}</div>}
          {doc?.synthetic && (
            <div className="note demo">DEMO DATA. This map shows synthetic patches until the real export is added
              (python -m deforest.web_export). The numbers above the map are real.</div>
          )}
          {stats && (
            <>
              <div className="kv">
                <span>Forest in 2000</span><b>{Math.round(stats.forest).toLocaleString()} ha</b>
                <span>Lost 2001–{Math.min(year, LAST)}</span>
                <b className="accent-rust">{Math.round(stats.lost).toLocaleString()} ha · {(100 * stats.lost / stats.forest).toFixed(1)}%</b>
                {stats.capture !== null && <>
                  <span>Loss in riskiest 10%</span><b className="accent-ochre">{(stats.capture * 100).toFixed(0)}%</b>
                </>}
              </div>
              <div>
                <div className="eyebrow" style={{ marginBottom: 8 }}>Forest lost per year</div>
                <div className="bars" aria-label="Forest lost per year">
                  {stats.perYear.map((v, k) => (
                    <i key={k} title={`${YEARS[k]}: ${Math.round(v)} ha`}
                       className={YEARS[k] === Math.min(year, LAST) && !fc ? "cur" : YEARS[k] <= Math.min(year, LAST) ? "on" : ""}
                       style={{ height: `${Math.max(2, (v / maxBar) * 100)}%` }} />
                  ))}
                </div>
              </div>
            </>
          )}
          <div className="legend">
            {fc ? (
              <>
                <div>Model risk of loss, 2020–23</div>
                <div className="ramp" />
                <div style={{ justifyContent: "space-between" }}><span>low</span><span>high</span></div>
                <div><i className="swatch" style={{ background: "#26302a" }} /> not scored (ambiguous label)</div>
                <label className="toggle">
                  <input type="checkbox" checked={showActual} onChange={(e) => setShowActual(e.target.checked)} />
                  outline patches that really lost forest
                </label>
              </>
            ) : (
              <>
                <div><i className="swatch" style={{ background: "#3f6340" }} /> forest standing</div>
                <div><i className="swatch" style={{ background: "#b9673a" }} /> part of the patch cleared</div>
                <div><i className="swatch" style={{ background: "#d0582c" }} /> heavily cleared</div>
              </>
            )}
          </div>
        </aside>
      </div>

      <div className="timeline">
        <div className="timeline-head">
          <div>
            <div className="eyebrow">{fc ? "Model forecast · made from early-2019 imagery" : "Observed · Hansen Global Forest Change"}</div>
            <div className={`year-big ${fc ? "forecast" : ""}`}>{fc ? "2020–23 forecast" : year}</div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ PAUSE" : "▶ PLAY"}</button>
            <button className="btn primary" onClick={() => { setPlaying(false); setYear(FORECAST); }}>SHOW FORECAST</button>
          </div>
        </div>
        <input className="slider" type="range" min={FIRST} max={FORECAST} step={1} value={year}
               aria-label="Year" onChange={(e) => { setPlaying(false); setYear(+e.target.value); }} />
        <div className="ticks">
          <span>2001</span><span>2006</span><span>2011</span><span>2016</span><span>2019</span><span>2023</span>
          <span className="fc">FORECAST</span>
        </div>
      </div>
    </div>
  );
}
