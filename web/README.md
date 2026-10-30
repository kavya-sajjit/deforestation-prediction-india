# canopy.watch: the project website

A static Next.js site: an earthy, OncoScan-style landing page with a year slider (Hansen forest loss
2001–2023) and the model's 2020–23 forecast on a map, plus the results and explanation checks.

## Run it on your laptop

```bash
cd web
npm install          # once
npm run dev          # opens on http://localhost:3000 and reloads as you edit
```

`npm run build` makes the production version in `web/out/` (plain HTML, CSS and JS);
`npm start` serves that folder on http://localhost:3000.

## Put real data in it

The map ships with **demo data** (a yellow banner says so). The numbers in the stats strip and the
results chart are real. To show the real maps, run section 6 of `notebooks/03_v2_colab.ipynb`
(or `python -m deforest.web_export --data data/v2 --results results --out web/public/data`),
unzip `website_data.zip` into `web/public/`, commit and push.

## Deploy

* **GitHub Pages (set up already):** in the repo's Settings → Pages, set Source to "GitHub Actions".
  Every push to `main` that changes `web/` rebuilds the site at
  `https://kavya-sajjit.github.io/deforestation-prediction-india/`.
* **Vercel (alternative, nicer URL):** import the repo on vercel.com, set Root Directory to `web`,
  and deploy. No settings beyond that.

No server or Docker is needed: the site is static files. A server only becomes necessary if you add
live features, such as running the model on a region the user picks.

## Files

| Path | What it is |
|---|---|
| `app/page.jsx` | the page: hero, stats, sections (reads `public/data/summary.json` at build time) |
| `components/MapExplorer.jsx` | MapLibre map, region tabs, year slider, forecast view |
| `components/AsciiTree.jsx` | the animated character tree in the hero |
| `components/Pipeline.jsx`, `Results.jsx` | how-it-works panel and model comparison |
| `app/globals.css` | colours, type and layout |
| `public/data/` | `summary.json` + one JSON per region (from `deforest.web_export`) |
