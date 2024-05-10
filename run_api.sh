#!/bin/bash

echo "Starting API"
source "./conda/etc/profile.d/conda.sh"
conda activate voicecraftapiconda
python3 VoiceCraftAPI.py