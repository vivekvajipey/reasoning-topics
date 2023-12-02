import itertools
import csv
import subprocess

iters_range = [100, 150, 200]
min_cf_range = [20, 30]
min_df_range = [10, 20]
rm_top_range = [25, 50]
num_topics_range = [30, 40, 50]
num_beta_sample_range = [1, 3, 5, 10]

# iters_range = [100]
# min_cf_range = [20, 30, 15]
# min_df_range = [20]
# rm_top_range = [50]
# num_topics_range = [30]
# num_beta_sample_range = [1]

hyperparameter_grid = itertools.product(
    iters_range, min_cf_range, min_df_range, rm_top_range, num_topics_range, num_beta_sample_range
)

def run_model(iters, min_cf, min_df, rm_top, num_topics, num_beta_sample):
    model_name = f'iters{iters}-min_cf{min_cf}-min_df{min_df}-rm_top{rm_top}-num_topics{num_topics}-num_beta_sample{num_beta_sample}'
    log_filename = f"logs/{model_name}.log"

    command = [
        'python', 'fit_topic_model.py', 
        '--iters', str(iters), 
        '--min_cf', str(min_cf), 
        '--min_df', str(min_df), 
        '--rm_top', str(rm_top), 
        '--num_topics', str(num_topics), 
        '--num_beta_sample', str(num_beta_sample)
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return 'crashed', str(e)

    try:
        with open(log_filename, 'r') as log_file:
            log_data = log_file.read()
        return 'finished', log_data
    except Exception as e:
        return 'crashed', f"Failed to read log file: {e}"

with open('hyperparameter_sweep_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['iters', 'min_cf', 'min_df', 'rm_top', 'num_topics', 'num_beta_sample', 'status', 'log_data'])

    for params in hyperparameter_grid:
        status, log_data = run_model(*params)
        writer.writerow([*params, status, log_data])