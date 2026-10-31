import fs from "node:fs";
import path from "node:path";
import AsciiTree from "@/components/AsciiTree";
import MapExplorer from "@/components/MapExplorer";
import Pipeline from "@/components/Pipeline";

function readSummary() {
  try {
    return JSON.parse(fs.readFileSync(path.join(process.cwd(), "public/data/summary.json"), "utf8"));
  } catch {
    return { tasks: {} };
  }
}

const GITHUB = "https://github.com/kavya-sajjit/deforestation-prediction-india";

export default function Page() {
  const s = readSummary();
  const rows = s.tasks?.forecast || [];
  const by = Object.fromEntries(rows.map((r) => [r.experiment, r]));
  const bcos = by.bcos_rgbn_hist;
  const patches = (s.dataset || []).reduce((a, d) => a + (d.patches || 0), 0);
  const f2 = (x) => (x == null ? "–" : x.toFixed(2));

  return (
    <>
      <nav className="nav">
        <div className="wrap">
          <a className="brand" href="#top">canopy<span>.</span>watch</a>
          <ul>
            <li><a href="#explore">Explore</a></li>
            <li><a href="#how">How it works</a></li>
            <li><a href="#limits">Limits</a></li>
            <li><a href={GITHUB}>Code</a></li>
          </ul>
        </div>
      </nav>

      <header className="hero" id="top">
        <div className="wrap">
          <div>
            <div className="eyebrow" style={{ marginBottom: 18 }}>Interpretable ML · Sentinel-2 · India</div>
            <h1>
              Four forests,<br />one satellite,<br />and where the trees <span className="accent-rust">go next.</span>
            </h1>
            <p className="lede">
              Forest loss is cheapest to stop before it happens. This model reads 2019 satellite imagery of four
              Indian forests and flags the patches most likely to be cleared over the next four years, and every
              flag comes with the exact pixels behind it.
            </p>
            <div className="cta">
              <a className="btn primary" href="#explore" style={{ borderBottom: "1px solid var(--moss)" }}>EXPLORE THE MAP</a>
              <a className="btn" href={GITHUB}>READ THE CODE</a>
            </div>
          </div>
          <AsciiTree />
        </div>
      </header>

      <div className="wrap">
        <div className="stats">
          <div className="stat"><b>{f2(bcos?.ap)}</b><span>forecast score</span></div>
          <div className="stat"><b>{f2(bcos?.chance_ap)}</b><span>random guessing</span></div>
          <div className="stat"><b>{bcos ? `${Math.round(bcos.capture_top10 * 100)}%` : "–"}</b><span>of real loss in its top 10% warnings</span></div>
          <div className="stat"><b>{patches ? patches.toLocaleString() : "4"}</b><span>{patches ? "forest squares · 4 regions" : "regions"}</span></div>
        </div>
      </div>

      <section className="block" id="explore">
        <div className="wrap">
          <div className="eyebrow">01 · Explore</div>
          <h2>Slide through two decades of clearing, then ask the model <span className="accent-ochre">what comes next.</span></h2>
          <MapExplorer />
        </div>
      </section>

      <section className="block" id="how">
        <div className="wrap">
          <div className="eyebrow">02 · How it works</div>
          <h2>From a cloudy satellite archive to a flag you can <span className="accent-moss">defend.</span></h2>
          <Pipeline />
        </div>
      </section>

      <section className="block" id="limits">
        <div className="wrap">
          <div className="eyebrow">03 · Limits</div>
          <h2>What this is not.</h2>
          <div className="limits">
            <div><b>Not field truth</b><p>Labels are Hansen's 30 m loss maps. They count shifting cultivation and plantation harvest as loss and miss slow degradation.</p></div>
            <div><b>Not all of India</b><p>Four 30 km regions. Two of them have few loss events, so their scores are noisy.</p></div>
            <div><b>Not a crystal ball</b><p>The forecast window is 2020–23 from 2019 imagery. A forecast past 2024 needs retraining on newer imagery.</p></div>
            <div><b>Not causal</b><p>Roads, mining leases, policy and prices drive clearing but are not inputs yet. Explanations show association, not cause.</p></div>
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap">
          <span>Built by Kavya Sajjit · D. J. Sanghvi College of Engineering, Mumbai</span>
          <span><a href={GITHUB}>GitHub</a> · Data: ESA Sentinel-2, Hansen/UMD/Google/USGS/NASA</span>
        </div>
      </footer>
    </>
  );
}
