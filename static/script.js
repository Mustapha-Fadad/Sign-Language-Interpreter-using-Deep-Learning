document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const startBtn = document.getElementById("start-btn");
    const stopBtn = document.getElementById("stop-btn");
    const videoStream = document.getElementById("video-stream");
    const videoPlaceholder = document.getElementById("video-placeholder");
    
    // UI Elements
    const predLabel = document.getElementById("pred-label");
    const predConf = document.getElementById("pred-conf");
    const confFill = document.getElementById("confidence-fill");
    const historyList = document.getElementById("history-list");
    const ttsToggle = document.getElementById("tts-toggle");

    // State Variables
    let isRunning = false;
    let pollInterval = null;
    let currentPrediction = "";
    let consecutiveCount = 0;
    
    // Audio Context
    let synth = window.speechSynthesis;

    function speak(text) {
        if (!ttsToggle.checked) return;
        if (synth.speaking) return;
        
        let utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.1; // Slightly faster
        synth.speak(utterance);
    }

    function updateHistory(label, confidence) {
        // Remove 'empty' placeholder if present
        const emptyItem = historyList.querySelector('.empty');
        if (emptyItem) emptyItem.remove();

        const li = document.createElement("li");
        li.className = "history-item";
        li.innerHTML = `<span class="label-text">${label}</span> <span class="conf-text">${Math.round(confidence)}%</span>`;
        
        // Add to top of list
        historyList.insertBefore(li, historyList.firstChild);

        // Keep history list short (max 6 items)
        if (historyList.children.length > 6) {
            historyList.removeChild(historyList.lastChild);
        }
    }

    function fetchPrediction() {
        if (!isRunning) return;

        fetch("/predict_feed")
            .then(res => res.json())
            .then(data => {
                const label = data.label;
                const conf = data.confidence;
                
                // Update text
                predLabel.innerText = label;
                predConf.innerText = `${conf}%`;
                
                // Update bar width & color
                confFill.style.width = `${conf}%`;
                
                if (label === "No Hand Detected" || label === "Camera Offline" || label === "Waiting...") {
                    confFill.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                    predLabel.style.color = 'var(--text-secondary)';
                    currentPrediction = "";
                    return;
                }

                if (conf > 75) {
                    confFill.style.backgroundColor = '#2ea043'; // Green
                    predLabel.style.color = '#58a6ff';
                } else if (conf > 50) {
                    confFill.style.backgroundColor = '#d29922'; // Yellow
                    predLabel.style.color = '#d29922';
                } else {
                    confFill.style.backgroundColor = '#f85149'; // Red
                    predLabel.style.color = '#f85149';
                }

                // Smooth history & TTS triggering logic
                // Only trigger if confidence is high and stable
                if (conf > 70) {
                    if (label === currentPrediction) {
                        consecutiveCount++;
                    } else {
                        currentPrediction = label;
                        consecutiveCount = 1;
                    }
                    
                    // Trigger if held for approx 10 polling cycles (~1 second at 100ms)
                    if (consecutiveCount === 10) {
                        speak(label);
                        updateHistory(label, conf);
                    }
                } else {
                    consecutiveCount = 0;
                }
            })
            .catch(err => console.error("Error fetching prediction:", err));
    }

    startBtn.addEventListener("click", () => {
        // Start streaming
        videoPlaceholder.style.display = "none";
        videoStream.style.display = "block";
        
        // Connect image tag to python generator
        videoStream.src = "/video_feed?" + new Date().getTime(); // prevent cache
        
        startBtn.style.display = "none";
        stopBtn.style.display = "block";
        isRunning = true;

        // Poll predictions 10 times a second
        pollInterval = setInterval(fetchPrediction, 100);
    });

    stopBtn.addEventListener("click", () => {
        // Graceful stop to backend API
        fetch("/stop_camera", { method: "POST" })
            .then(() => {
                videoStream.src = "";
                videoStream.style.display = "none";
                videoPlaceholder.style.display = "flex";
                
                startBtn.style.display = "block";
                stopBtn.style.display = "none";
                
                clearInterval(pollInterval);
                isRunning = false;
                
                predLabel.innerText = "Camera Offline";
                predLabel.style.color = 'var(--text-secondary)';
                confFill.style.width = '0%';
                predConf.innerText = "0%";
            });
    });
});
