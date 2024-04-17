import os.path

from plots import plot_main

experiments_to_evaluate = [
    '10k_bpr_rank_based',
    '10k_bpr_us_centric',
    '100k_sample1_bpr_rank_based',
    '100k_sample1_neumf_rank_based',
    '100k_sample1_itemknn_rank_based',
    '100k_sample1_multivae_rank_based',
    # add more experiments here
]

if __name__ == '__main__':
    for experiment in experiments_to_evaluate:
        if not os.path.exists(os.path.join('experiments', experiment)):
            print(f"ERROR: Skipping {experiment}: Experiment does not exist!")
            continue
        print(f'Processing experiment "{experiment}"')
        plot_main(experiment_name=experiment, force_recompute=True)