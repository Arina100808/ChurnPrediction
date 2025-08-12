#!/bin/bash

docker run --rm -v "$PWD":/app churn-prediction predict \
  --data src/data/data.csv \
  --model artifacts/default/model_default.joblib
