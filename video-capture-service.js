// Select DOM elements
const video = document.getElementById('video-stream');
const canvas = document.getElementById('capture-canvas');
const captureButton = document.getElementById('capture-frame');
const startButton = document.getElementById('start-video');
const stopButton = document.getElementById('stop-video');

let stream = null; // To store the media stream
let captureInterval = null; // For repeated frame capturing
let captureTimeout = null; // To track the 30-second timeout

// Start video stream
startButton.addEventListener('click', async () => {
    try {
        // Request access to the user's webcam
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        console.log("Video stream started.");
    } catch (error) {
        console.error("Error accessing video stream:", error);
        alert("Could not access the webcam. Please check your permissions.");
    }
});

// Stop video stream
stopButton.addEventListener('click', () => {
    try {
        stopCapturing(); // Stop capturing frames
        if (stream) {
            // Stop all media tracks
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            console.log("Video stream stopped.");
        } else {
            console.warn("No active video stream to stop.");
        }
    } catch (error) {
        console.error("Error occurred while stopping the video stream:", error);
        alert("An error occurred while trying to stop the video stream. Please try again.");
    }
});

// Capture frame and send to the server
captureButton.addEventListener('click', () => {
    // Clear existing intervals or timeouts if any
    if (captureInterval) clearInterval(captureInterval);
    if (captureTimeout) clearTimeout(captureTimeout);

    // Start capturing frames every second
    captureInterval = setInterval(captureAndSendFrame, 1000); // 1000ms = 1 second

    // Automatically stop capturing after 30 seconds
    captureTimeout = setTimeout(() => {
        stopCapturing();
        console.log("Stopped capturing after 30 seconds.");
    }, 30000); // 30000ms = 30 seconds
});

// Function to capture and send a frame
function captureAndSendFrame() {
    // Create a new canvas element for each capture
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 640;
    tempCanvas.height = 480;
    const context = tempCanvas.getContext('2d');

    // Draw the current video frame onto the canvas
    context.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    // Convert canvas image to Base64
    const base64Image = tempCanvas.toDataURL('image/jpeg');

    // Prepare the JSON payload
    const jsonPayload = {
        image: base64Image,
        process_duration: 30,
        frame_rate: 1, // 1 frame per second
        dimensions: {
            width: tempCanvas.width,
            height: tempCanvas.height,
        },
    };

    // Send the payload to the server
    fetch('https://dataanalyst.pt/faceportal/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(jsonPayload),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Server response:", data);
        })
        .catch(error => {
            console.error("Error sending data to the server:", error);
        });
}

// Function to stop capturing frames
function stopCapturing() {
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    if (captureTimeout) {
        clearTimeout(captureTimeout);
        captureTimeout = null;
    }
    console.log("Frame capturing stopped.");
}
