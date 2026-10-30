/** Static export: `npm run build` writes plain HTML/JS to ./out, which any static host serves.
 *  For GitHub Pages under /<repo>, set NEXT_PUBLIC_BASE_PATH=/deforestation-prediction-india */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";
export default {
  output: "export",
  basePath,
  images: { unoptimized: true },
  trailingSlash: true,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};
