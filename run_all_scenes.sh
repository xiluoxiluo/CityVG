#!/bin/bash
set -e

SCENES=(
  # birmingham_block_1
  # birmingham_block_4
  birmingham_block_9
)

for SCENE_ID in "${SCENES[@]}"; do
  bash run_scene.sh "$SCENE_ID"
done
