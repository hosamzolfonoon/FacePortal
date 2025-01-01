#!/bin/bash
set -x  # Enable script debugging
echo "Copying default configuration to Nginx sites-available..."
cp -rf default /etc/nginx/sites-available/default

echo "Reloading Nginx..."
nginx -s reload

echo "Starting Gunicorn server..."
gunicorn -w 3 --bind 0.0.0.0:8000 --daemon faceportal:app