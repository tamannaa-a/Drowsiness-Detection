import React, { useRef, useEffect, useState } from "react";

/*
  FRONTEND NOTES:
  - Configure backend URL via REACT_APP_BACKEND_URL env variable.
  - The frontend captures a frame from the webcam every INTERVAL_MS and posts it as multipart/form-data to /predict
  - Frontend uses consecutive closed-frame counts to reduce flicker.
*/

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000/predict";
const INTERVAL_MS = 700;        // how often to send frames
const CLOSED_THRESHOLD = 0.6;   // backend score threshold to count as "closed"
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

  // Web Audio API beep generator (no external file)
  function beep(duration = 200, frequency = 800, volume = 0.2) {
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
    // request webcam on mount
    async function initWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
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
  }, [running]);

  async function captureAndSend() {
    if (!videoRef.current || videoRef.current.readyState < 2) {
      return;
    }
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const w = 320, h = 240;
    canvas.width = w; canvas.height = h;
    ctx.drawImage(video, 0, 0, w, h);

    setStatus("Sending...");
    canvas.toBlob(async (blob) => {
      if (!blob) { setStatus("Capture failed"); return; }

      // prepare multipart
      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      try {
        const resp = await fetch(BACKEND_URL, { method: "POST", body: form });
        if (!resp.ok) {
          setStatus("Server error");
          return;
        }
        const data = await resp.json();
        setLabel(data.label);
        setScore(Number(data.score).toFixed(2));
        setStatus("OK");

        // check closed score
        if (data.label === "Closed" && data.score >= CLOSED_THRESHOLD) {
          closedCountRef.current += 1;
        } else {
          closedCountRef.current = 0;
          setAlertState(false);
        }

        // if closed for enough consecutive frames -> alert
        if (closedCountRef.current >= CLOSED_CONSEC) {
          if (!alertState) {
            setAlertState(true);
            // beep and visual
            beep(350, 750, 0.2);
          }
        }
      } catch (e) {
        console.error(e);
        setStatus("Network error");
      }
    }, "image/jpeg", 0.85);
  }

  return (
    <div style={{ fontFamily: "Arial,Helvetica,sans-serif", padding: 20 }}>
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

      <div style={{ marginTop: 18 }}>
        <strong>Notes:</strong> If the browser blocks audio autoplay, interact with the page (click start) to allow beep playback.
      </div>
    </div>
  );
}

export default App;
