const video = document.getElementById('video-stream');
const canvas = document.getElementById('capture-canvas');
const captureButton = document.getElementById('capture-frame');
const startButton = document.getElementById('start-video');
const stopButton = document.getElementById('stop-video');

let stream;
let captureInterval;
let captureTimeout; // To track the 30-second timeout

// Start video
startButton.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error("Error accessing video stream:", error);
    }
});

// Stop video
stopButton.addEventListener('click', () => {
    stopCapturing(); // Call the stopCapturing function to clear intervals and timeouts
    try {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        } else {
            console.warn("No active video stream to stop.");
        }
    } catch (error) {
        console.error("Error stopping video stream:", error);
    }
});

// Capture frame and send to the server
captureButton.addEventListener('click', () => {
    if (captureInterval) {
        clearInterval(captureInterval); // Clear any existing interval
    }
    if (captureTimeout) {
        clearTimeout(captureTimeout); // Clear any existing timeout
    }

    // Start capturing frames every second
    captureInterval = setInterval(() => {
        captureAndSendFrame();
    }, 1000); // 1000ms = 1 second

    // Stop capturing automatically after 30 seconds
    captureTimeout = setTimeout(() => {
        stopCapturing();
        console.log("Stopped capturing after 30 seconds.");
    }, 30000); // 30000ms = 30 seconds
});

// Function to capture and send a frame
function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to Base64
    const base64Image = canvas.toDataURL('image/jpeg');

    // Create JSON payload
    const jsonPayload = {
        image: base64Image,
        timestamp: Date.now(),
        process_duration: 30,
        frame_rate: 1, // 1 frame per second
        detection_threshold: 30,
        dimensions: {
            width: canvas.width,
            height: canvas.height,
        },
    };

    // Send JSON payload to the server
    fetch('https://dataanalyst.pt/ferapp/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(jsonPayload),
    })
    .then((response) => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then((data) => {
        console.log("Server response:", data);
    })
    .catch((error) => {
        console.error("Error sending data to the server:", error);
    });
}

// Function to stop capturing
function stopCapturing() {
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    if (captureTimeout) {
        clearTimeout(captureTimeout);
        captureTimeout = null;
    }
    console.log("Capturing stopped.");
}
