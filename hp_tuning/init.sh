#!/bin/bash

CACHE_DIR=/tmp/DeBERTa/
cd ../experiments/glue
./download_data.sh  $CACHE_DIR/glue_tasks