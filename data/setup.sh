#!/bin/bash

# List of URLs
URLS=(
    "https://www.fema.gov/about/reports-and-data/openfema/FimaNfipClaims.csv"
    "https://www.fema.gov/about/reports-and-data/openfema/FimaNfipPolicies.csv"
)

# Loop through each URL
for URL in "${URLS[@]}"; do
    FILENAME=$(basename "$URL")
    
    if [ -f "$FILENAME" ]; then
        echo "File '$FILENAME' already exists. Skipping download."
    else
        echo "Downloading '$FILENAME'..."
        curl -O "$URL"
    fi
done
