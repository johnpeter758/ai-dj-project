#!/bin/bash
# Download chart songs script
# Run with: bash download_songs.sh

# Install yt-dlp with Python 3.11 if needed
if ! command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    brew install python@3.11
fi

# Create music directories
mkdir -p ~/Music/ChartHits ~/Music/JohnSummit ~/Music/Drake

echo "Downloading chart songs..."
echo "Using: /opt/homebrew/bin/python3.11 -m yt_dlp"

# You may need to:
# 1. Connect to Proton VPN
# 2. Or use a browser extension to export cookies
# 3. Or download manually from YouTube

echo ""
echo "To download songs, you can also:"
echo "1. Use a browser with YouTube Video Downloader extension"
echo "2. Use 4K Video Downloader app"
echo "3. Use https://y2mate.com web service"
echo ""
echo "Then copy downloaded files to:"
echo "  - ~/Music/ChartHits/"
echo "  - ~/Music/JohnSummit/"
echo "  - ~/Music/Drake/"
