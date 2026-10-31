import fs from "node:fs";
import path from "node:path";
import AsciiTree from "@/components/AsciiTree";
import MapExplorer from "@/components/MapExplorer";
import Pipeline from "@/components/Pipeline";
import Results from "@/components/Results";
import { BASE } from "@/lib/data";

function readSummary() {
  try {
    return JSON.parse(fs.readFileSync(path.join(process.cwd(), "public/data/summary.json"), "utf8"));
  } catch {
    return { tasks: {} };
  }
}
const hasImg = (f) => fs.existsSync(path.join(process.cwd(), "public/img", f));

const GITHUB = "https://github.com/kavya-sajjit/deforestation-prediction-india";

export default function Page() {
  const s = readSummary();
  const rows = s.tasks?.forecast || [];
  const by = Object.fromEntries(rows.map((r) => [r.experiment, r]));
  const bcos = by.bcos_rgbn_hist, gbm = by.gbm, cnn = by.cnn_rgbn_hist, hist = by.history_lr;
  const patches = (s.dataset || []).reduce((a, d) => a + (d.patches || 0), 0);
  const folds = bcos?.splits;
  const f2 = (x) => (x == null ? "–" : x.toFixed(2));

  return (
    <>
      <nav className="nav">
        <div className="wrap">
          <a className="brand" href="#top">canopy<span>.</span>watch</a>
          <ul>
            <li><a href="#explore">Explore</a></li>
            <li><a href="#how">How it works</a></li>
            <li><a href="#results">Results</a></li>
            <li><a href="#explain">Explanations</a></li>
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
          <div className="stat"><b>{f2(bcos?.ap)}</b><span>PR-AUC, interpretable model</span></div>
          <div className="stat"><b>{f2(bcos?.chance_ap)}</b><span>chance level</span></div>
          <div className="stat"><b>{bcos ? `${Math.round(bcos.capture_top10 * 100)}%` : "–"}</b><span>of loss in riskiest 10%</span></div>
          <div className="stat"><b>{patches ? patches.toLocaleString() : "4"}</b><span>{patches ? "forest patches · 4 biomes" : "biomes"}</span></div>
        </div>
        {s.note && <p className="eyebrow" style={{ marginTop: 12, textTransform: "none", letterSpacing: ".04em" }}>{s.note}</p>}
      </div>

      <section className="block" id="explore">
        <div className="wrap">
          <div className="eyebrow">01 · Explore</div>
          <h2>Slide through two decades of clearing, then ask the model <span className="accent-ochre">what comes next.</span></h2>
          <p className="intro">
            Each square is a 640 m forest patch. Drag the slider to watch Hansen's yearly loss maps accumulate from
            2001 to 2023. The last stop shows the model's forecast, made only from early-2019 imagery and
            the loss history before it, with outlines on the patches that really were cleared in 2020–23.
          </p>
          <MapExplorer />
        </div>
      </section>

      <section className="block" id="how">
        <div className="wrap">
          <div className="eyebrow">02 · How it works</div>
          <h2>From a cloudy satellite archive to a flag you can <span className="accent-moss">defend.</span></h2>
          <p className="intro">
            Labels come from an independent source (Hansen Global Forest Change), never from the model's own
            pixels, and every score is measured on whole regions the model never saw during training.
          </p>
          <Pipeline />
        </div>
      </section>

      <section className="block" id="results">
        <div className="wrap">
          <div className="eyebrow">03 · Results</div>
          <h2>Interpretable, and honest about what that costs.</h2>
          <p className="intro">
            Average precision for forecasting 2020–23 loss{folds ? `, averaged over ${folds} spatially held-out folds` : ""}.
            Rare events make accuracy meaningless here, so the bar to beat is chance ({f2(bcos?.chance_ap)}) and a
            model that only knows where clearing already happened ({f2(hist?.ap)}).
          </p>
          <Results rows={rows} />
          <div className="callouts">
            <div>
              <h3 className="accent-moss">It forecasts</h3>
              <p>The interpretable model scores {f2(bcos?.ap)} against a chance level of {f2(bcos?.chance_ap)}, and
                 beats the history-only baseline ({f2(hist?.ap)}).</p>
            </div>
            <div>
              <h3 className="accent-ochre">Interpretability has a price</h3>
              <p>A black-box CNN reaches {f2(cnn?.ap)} and gradient boosting {f2(gbm?.ap)}. Closing that gap without
                 giving up exact explanations is the open problem.</p>
            </div>
            <div>
              <h3 className="accent-rust">Near-infrared is not free</h3>
              <p>None of the three ways of adding NIR to an ImageNet-pretrained B-cos model beats colour alone yet,
                 while NIR does help models trained from scratch.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="block" id="explain">
        <div className="wrap">
          <div className="eyebrow">04 · Explanations</div>
          <h2>Every flag adds up, exactly.</h2>
          <p className="intro">
            B-cos networks (Böhle et al., CVPR 2022) have no biases and compute a dynamic linear map, so the evidence
            for "loss" is an exact sum over the input pixels. The project tests it: contributions sum to the
            decision to within 10⁻⁶, they are compared with Grad-CAM on a black-box model, and checked against a
            model with random weights.
          </p>
          <div className="formula">
            logit<sub>loss</sub>(x) − logit<sub>no loss</sub>(x) = Σ<sub>c,h,w</sub>&nbsp; <em>W(x)</em><sub>c,h,w</sub> · [x, 1−x]<sub>c,h,w</sub>
          </div>
          {(hasImg("gallery_forecast.png") || hasImg("gallery_detect.png")) && (
            <div className="gallery">
              {hasImg("gallery_detect.png") && (
                <figure><img src={`${BASE}/img/gallery_detect.png`} alt="Satellite patches, Hansen loss, and B-cos evidence maps for detected loss" />
                  <figcaption>Detection: evidence lands on the cleared pixels.</figcaption></figure>
              )}
              {hasImg("gallery_forecast.png") && (
                <figure><img src={`${BASE}/img/gallery_forecast.png`} alt="Satellite patches, later Hansen loss, and B-cos evidence maps for the forecast" />
                  <figcaption>Forecast: evidence from the 2019 imagery for loss that came later.</figcaption></figure>
              )}
            </div>
          )}
        </div>
      </section>

      <section className="block" id="limits">
        <div className="wrap">
          <div className="eyebrow">05 · Limits</div>
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
