"use client";
import { useEffect, useRef, useState } from "react";

const STEPS = [
  ["01", "Satellite photos, early 2019", "Sentinel-2"],
  ["02", "Cut into 640 m squares", "7,176"],
  ["03", "Add where trees were already cut", "2001–18"],
  ["04", "Predict which squares lose forest next", "2020–23"],
  ["05", "Show the pixels behind each warning", "B-cos"],
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
      {STEPS.map(([n, t, s], i) => (
        <div key={n} className={`step ${i < on ? "done" : ""}`}>
          <span className="n">[{n}]</span>
          <span>{t}</span>
          <span className="meter"><i style={{ width: i < on ? "100%" : "0%" }} /></span>
          <span className="s">{i < on ? s : "waiting"}</span>
        </div>
      ))}
    </div>
  );
}
