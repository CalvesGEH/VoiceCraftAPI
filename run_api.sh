#!/bin/bash

echo "Starting API"
./conda/bin/conda init
./conda/bin/conda activate voicecraftapi
python3 VoiceCraftAPI.py