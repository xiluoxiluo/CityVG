#!/bin/bash
set -e

SCENE_ID=$1

if [ -z "$SCENE_ID" ]; then
  echo "Usage: bash run_scene.sh <SCENE_ID>"
  exit 1
fi

echo "==============================="
echo "Running pipeline for scene: $SCENE_ID"
echo "==============================="

python preprocess/saveBlockJson.py   --SCENE_ID "$SCENE_ID"
python preprocess/drawSemMap.py      --SCENE_ID "$SCENE_ID"
python preprocess/genInsCenter.py    --SCENE_ID "$SCENE_ID"
python preprocess/concatImage.py     --SCENE_ID "$SCENE_ID"

python sceneGraph/genGraph.py        --SCENE_ID "$SCENE_ID"
python sceneGraph/genCaption.py     --SCENE_ID "$SCENE_ID"
python sceneGraph/graphClustering.py --SCENE_ID "$SCENE_ID"
python sceneGraph/contraGraph.py     --SCENE_ID "$SCENE_ID"

python sceneGraph/graphInference_Doubao.py --SCENE_ID "$SCENE_ID"

echo "Finished scene: $SCENE_ID"
