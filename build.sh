#!/bin/bash

uv run jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --to python backend.ipynb
