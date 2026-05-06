#!/bin/bash
# One-command launcher for Raw2Drive + OAIAD

set -e
source ~/miniconda3/bin/activate raw2drive

# Start CARLA (headless)
echo "[CARLA] Starting server..."
tmux new-session -d -s carla "cd ~/carla && ./CarlaUE4.sh -RenderOffScreen -carla-server -fps=20 -quality-level=Low"
sleep 15

# Test CARLA connection
python3 -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10); print('[CARLA] v'+c.get_server_version())"

# Train
echo "[Training] Starting..."
cd ~/raw2drive_oaiad
python training/train.py \
  --stage all \
  --batch_size 8 \
  --steps 50000 \
  "$@"

echo "[Done] Training complete!"
