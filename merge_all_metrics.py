import os

import pandas as pd

experiments_to_evaluate = [
    '10k_bpr_us_centric',
    '100k_sample1_bpr_rank_based',
    '100k_sample1_neumf_rank_based',
    '100k_sample1_itemknn_rank_based',
    '100k_sample1_multivae_rank_based',
    # add more experiments here
]

if __name__ == '__main__':
    metrics = []
    for experiment in experiments_to_evaluate:
        if not os.path.exists(os.path.join('experiments', experiment)):
            print(f"ERROR: Skipping {experiment}: Experiment does not exist!")
            continue
        # load metrics from file
        met = pd.read_csv(os.path.join('experiments', experiment, 'metrics.csv'))
        metrics.append(met)

    # merge all metrics into a single DataFrame
    all_metrics = pd.concat(metrics, ignore_index=True)
    # save merged metrics to file
    all_metrics.to_csv('metrics_merged.csv', index=False)