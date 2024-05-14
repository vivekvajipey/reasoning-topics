# !/bin/bash

# Set initial values
START_ROW=0
NUM_ROWS=1
NUM_SAMPLES=10


echo "Processing started."

# Generate answers
python generate_gsm8k_answers_from_instruct.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES --verbose
python generate_gsm8k_answers_from_instruct.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES --direct_prompt --verbose

# Calculate entropy
python calculate_gsm8k_entropy_from_base.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES
python calculate_gsm8k_entropy_from_base.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES --direct_prompt

echo "Processing complete."

# !/bin/bash

# # Set initial values
# START_ROW=0
# NUM_ROWS=50
# NUM_SAMPLES=25


# # Number of iterations, adjust as needed
# NUM_ITERATIONS=28

# for (( i=0; i<NUM_ITERATIONS; i++ ))
# do
#     echo "Processing from row $START_ROW to $(($START_ROW + $NUM_ROWS - 1))"

#     # Generate answers
#     python generate_gsm8k_answers_from_instruct.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES
#     python generate_gsm8k_answers_from_instruct.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES --direct_prompt

#     # Calculate entropy
#     python calculate_gsm8k_entropy_from_base.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES
#     python calculate_gsm8k_entropy_from_base.py --start_row=$START_ROW --num_rows=$NUM_ROWS --num_samples=$NUM_SAMPLES --direct_prompt

#     # Increment start_row for next iteration
#     START_ROW=$(($START_ROW + $NUM_ROWS))
# done

# echo "Processing complete."


