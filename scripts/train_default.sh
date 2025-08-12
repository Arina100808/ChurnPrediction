#!/bin/bash

docker run --rm -v "$PWD":/app churn-prediction \
  train --data src/data/data.csv
