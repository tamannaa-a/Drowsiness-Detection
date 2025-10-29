// frontend/src/App.js
import React, { useRef, useEffect, useState } from "react";

/*
  FRONTEND:
  - captures webcam frames
  - sends a snapshot to backend at interval
  - uses consecutive-closed-frame logic to raise alert
  - set BACKEND_URL using REACT_APP_BACKEND_URL env variable or default to localhost
*/

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000/predict";
const INTERVAL_MS = 700;        // interval between frames (ms)
const CLOSED_THRESHOLD = 0.6;   // backend score threshold to consider "Closed"
const CLOSED_CONSEC = 3;        // consecutive frames to trigger alert

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [label, setLabel] = useState("N/A");
  const [score, setScore] = useState(0);
  const closedCountRef = useRef(0);
  const [alertState, setAlertState] = useState(false);

  // beep generator
  function beep(duration = 300, frequency = 750, volume = 0.2) {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.type = "sine";
      o.frequency.value = frequency;
      g.gain.value = volume;
      o.connect(g);
      g.connect(ctx.destination);
      o.start(0);
      setTimeout(() => { o.stop(); ctx.close(); }, duration);
    } catch (e) {
      console.warn("Beep failed:", e);
    }
  }

  useEffect(() => {
    async function initWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        setStatus("Webcam access denied");
      }
    }
    initWebcam();
  }, []);

  useEffect(() => {
    let timer = null;
    if (running) {
      timer = setInterval(() => {
        captureAndSend();
      }, INTERVAL_MS);
    } else {
      if (timer) clearInterval(timer);
    }
    return () => { if (timer) clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  async function captureAndSend() {
    if (!videoRef.current || videoRef.current.readyState < 2) {
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const w = 320, h = 240;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);

    setStatus("Sending...");
    canvas.toBlob(async (blob) => {
      if (!blob) { setStatus("Capture failed"); return; }

      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      try {
        const resp = await fetch(BACKEND_URL, { method: "POST", body: form });
        if (!resp.ok) {
          setStatus("Server error: " + resp.status);
          return;
        }
        const data = await resp.json();
        setLabel(data.label);
        setScore(Number(data.score).toFixed(2));
        setStatus("OK");

        if (data.label === "Closed" && data.score >= CLOSED_THRESHOLD) {
          closedCountRef.current += 1;
        } else {
          closedCountRef.current = 0;
          setAlertState(false);
        }

        if (closedCountRef.current >= CLOSED_CONSEC) {
          if (!alertState) {
            setAlertState(true);
            beep(400, 800, 0.2);
          }
        }
      } catch (e) {
        console.error(e);
        setStatus("Network error");
      }
    }, "image/jpeg", 0.85);
  }

  return (
    <div style={{ padding: 18 }}>
      <h2>🚗 Emotion-Aware Driving — Eye-State Detector</h2>

      <div style={{ display: "flex", gap: 20 }}>
        <div>
          <video ref={videoRef} autoPlay playsInline muted style={{ width: 480, height: 360, borderRadius: 6, border: "1px solid #ccc" }} />
          <canvas ref={canvasRef} style={{ display: "none" }} />
          <div style={{ marginTop: 8 }}>
            <button onClick={() => setRunning(r => !r)} style={{ padding: "8px 12px" }}>
              {running ? "Stop Monitoring" : "Start Monitoring"}
            </button>
          </div>
        </div>

        <div style={{ width: 360 }}>
          <h3>Status: {status}</h3>
          <p>Label: <b>{label}</b></p>
          <p>Score: <b>{score}</b></p>

          <div style={{
            marginTop: 12,
            padding: 12,
            borderRadius: 8,
            background: alertState ? "#ffdddd" : "#ddffdd",
            border: "1px solid #aaa"
          }}>
            <h4 style={{ margin: 0 }}>
              {alertState ? "⚠️ DROWSINESS WARNING — WAKE UP!" : "Driver Attentive"}
            </h4>
            <p style={{ marginTop: 8, color: "#333", fontSize: 14 }}>
              Consecutive closed frames: {closedCountRef.current}
            </p>
          </div>

          <div style={{ marginTop: 20 }}>
            <small>
              Backend: <code>{BACKEND_URL}</code><br />
              Send interval: {INTERVAL_MS} ms • Closed threshold: {CLOSED_THRESHOLD} • Consecutive frames to alert: {CLOSED_CONSEC}
            </small>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
