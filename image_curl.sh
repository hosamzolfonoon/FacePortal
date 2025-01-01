#!/bin/bash

# Path to the image file
IMAGE_PATH="/home/hosam/Projects/FacePortal/01.jpg"

# Convert the image to Base64
BASE64_IMAGE=$(base64 -w 0 "$IMAGE_PATH")

# Create a temporary file for the JSON payload
JSON_PAYLOAD=$(mktemp)

# Construct the JSON payload and write it to the temporary file
cat <<EOF > "$JSON_PAYLOAD"
{
  "image": "data:image/jpeg;base64,$BASE64_IMAGE",
  "frame_rate": 1,
  "dimensions": {
    "width": 640,
    "height": 480
  }
}
EOF

# Use curl to send the payload, reading from the file
curl -X POST https://dataanalyst.pt/faceportal/ \
-H "Content-Type: application/json" \
-d @"$JSON_PAYLOAD"

# Clean up the temporary file
rm "$JSON_PAYLOAD"