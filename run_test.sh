#!/bin/bash

curl -X POST -H "Content-Type: application/json" -d '{"x": 5}' http://localhost:8000/predict
