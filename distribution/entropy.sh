#!/bin/bash

START_ROW=4
NUM_ROWS=5
NUM_SAMPLES=10

python generate_gsm8k_answers_from_instruct.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES

python calculate_gsm8k_entropy_from_base.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES