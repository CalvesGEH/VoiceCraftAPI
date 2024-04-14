#!/bin/bash

echo "Starting API"
source "./conda/etc/profile.d/conda.sh"
conda activate voicecraftapi
python3 VoiceCraftAPI.py