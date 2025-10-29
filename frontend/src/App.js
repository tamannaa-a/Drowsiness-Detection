import React, { useRef, useState, useEffect } from "react";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [alertMsg, setAlertMsg] = useState("");
  const [isDrowsy, setIsDrowsy] = useState(false);
  const [running, setRunning] = useState(false);

  const backendURL = "https://drowsiness-detection-1-djgk.onrender.com/predict";

  useEffect(() => {
    startCamera();
  }, []);

  // Start webcam
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      videoRef.current.srcObject = stream;
    } catch (error) {
      console.error("Error accessing webcam:", error);
    }
  };

  // Play alert sound
  const playAlertSound = () => {
    const audio = new Audio(
      "https://actions.google.com/sounds/v1/alarms/beep_short.ogg"
    );
    audio.play();
  };

  // Capture frame and send to backend
  const detectDrowsiness = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      try {
        const response = await fetch(backendURL, {
          method: "POST",
          body: formData,
        });
        const data = await response.json();

        if (data.prediction === "closed") {
          setAlertMsg("⚠️ Drowsiness Detected! Please stay alert!");
          if (!isDrowsy) playAlertSound();
          setIsDrowsy(true);
        } else {
          setAlertMsg("✅ Eyes Open — You’re alert!");
          setIsDrowsy(false);
        }
      } catch (error) {
        console.error("Prediction error:", error);
      }
    }, "image/jpeg");
  };

  // Start/stop detection loop
  useEffect(() => {
    let interval;
    if (running) {
      interval = setInterval(detectDrowsiness, 2000); // every 2 sec
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [running]);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🚗 Real-Time Drowsiness Detection System</h1>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        width="640"
        height="480"
        style={styles.video}
      />

      <canvas ref={canvasRef} width="640" height="480" style={{ display: "none" }} />

      <div style={styles.buttonContainer}>
        {!running ? (
          <button onClick={() => setRunning(true)} style={styles.startBtn}>
            ▶️ Start Detection
          </button>
        ) : (
          <button onClick={() => setRunning(false)} style={styles.stopBtn}>
            ⏹ Stop Detection
          </button>
        )}
      </div>

      <div style={styles.alertBox(isDrowsy)}>
        <p>{alertMsg}</p>
      </div>

      <footer style={styles.footer}>
        Built by <b>Tamanna Vaikkath</b> | Emotion-Aware Driver Safety System
      </footer>
    </div>
  );
}

// CSS Styles
const styles = {
  container: {
    textAlign: "center",
    backgroundColor: "#f9fafb",
    minHeight: "100vh",
    padding: "20px",
  },
  title: {
    fontSize: "2rem",
    color: "#333",
    marginBottom: "20px",
  },
  video: {
    borderRadius: "10px",
    boxShadow: "0 4px 10px rgba(0,0,0,0.2)",
  },
  buttonContainer: {
    marginTop: "20px",
  },
  startBtn: {
    backgroundColor: "#4CAF50",
    color: "white",
    border: "none",
    padding: "10px 20px",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "1rem",
  },
  stopBtn: {
    backgroundColor: "#e74c3c",
    color: "white",
    border: "none",
    padding: "10px 20px",
    borderRadius: "8px",
    cursor: "pointer",
    fontSize: "1rem",
  },
  alertBox: (isDrowsy) => ({
    marginTop: "30px",
    padding: "15px",
    backgroundColor: isDrowsy ? "#ffcccc" : "#ccffcc",
    borderRadius: "8px",
    width: "60%",
    margin: "auto",
    fontWeight: "bold",
    color: "#333",
  }),
  footer: {
    marginTop: "40px",
    color: "#666",
  },
};

export default App;
