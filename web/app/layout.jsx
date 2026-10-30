import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

export const metadata = {
  title: "Canopy Watch · Forecasting forest loss in India",
  description:
    "An interpretable model that reads Sentinel-2 imagery of four Indian forests and flags where trees are likely to be cleared next, with the pixels behind every flag.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
