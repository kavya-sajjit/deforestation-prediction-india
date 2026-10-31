"use client";
import { useEffect, useRef, useState } from "react";

const STEPS = [
  ["01", "Sentinel-2 composite", "Jan–Apr 2019, cloud-masked median · R, G, B, NIR at 10 m", "4 bands"],
  ["02", "Patch grid", "640 m forest patches, non-overlapping, ≥30% forest at the start of 2019", "7,176"],
  ["03", "Loss history", "Hansen Global Forest Change, 2001–2018: where clearing already happened", "18 yrs"],
  ["04", "B-cos forecast", "Interpretable network: the decision is an exact sum of per-pixel evidence", "exact"],
  ["05", "Explanation", "Per-pixel and per-band contributions for every flagged patch", "per pixel"],
  ["06", "Spatial check", "Whole 5 km blocks held out; scored against real 2020–23 loss", "5 folds"],
];

export default function Pipeline() {
  const ref = useRef(null);
  const [on, setOn] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let timer;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        let k = 0;
        timer = setInterval(() => { k += 1; setOn(k); if (k >= STEPS.length) clearInterval(timer); }, 420);
        io.disconnect();
      }
    }, { threshold: 0.35 });
    io.observe(el);
    return () => { io.disconnect(); clearInterval(timer); };
  }, []);

  return (
    <div className="pipeline" ref={ref}>
      {STEPS.map(([n, t, d, s], i) => (
        <div key={n} className={`step ${i < on ? "done" : ""}`}>
          <span className="n">[{n}]</span>
          <span>{t}<span className="d">{d}</span></span>
          <span className="meter"><i style={{ width: i < on ? "100%" : "0%" }} /></span>
          <span className="s">{i < on ? s : "waiting"}</span>
        </div>
      ))}
    </div>
  );
}
