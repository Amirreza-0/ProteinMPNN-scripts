# script to calculate the KL divergence between two probability distributions


# imports
import numpy as np

# main function
def calculate_kl_divergence(output_dir, unconditional_only=False, conditional_only=False):
    """
    Calculate the Kullback-Leibler divergence between two sets of log probabilities.

    Args:
        input_log_probs (np.ndarray): Log probabilities of the input distribution.
        benchmark_log_probs (np.ndarray): Log probabilities of the benchmark distribution.

    Returns:
        float: Kullback-Leibler divergence between the two distributions.
    """
    print(output_dir)
    output_dir_original = output_dir.replace(output_dir.split("/")[-2], output_dir.split("/")[-2] + "_original")
    output_dir_original = output_dir_original.replace(output_dir.split("/")[-1], output_dir.split("/")[-1].split("_")[0])
    print(output_dir_original)
    # 1.(split by / select first part), 2.(split by _ select first part) and replace this part with 1.
    # output_dir_original = output_dir_original.replace(output_dir_original.split('/')[0], output_dir_original.split('/')[0].split('_')[0])


    # log probabilities path
    if unconditional_only:
        log_probs_original_path = f"{output_dir_original}/unconditional_probs_only/*.npz"
        log_probs_new_path = f"{output_dir}/unconditional_probs_only/*.npz"
        save_path = f"{output_dir}/unconditional_probs_only/kl_divergence.npy"
    elif conditional_only:
        log_probs_original_path = f"{output_dir_original}/conditional_probs_only*.npz"
        log_probs_new_path = f"{output_dir}/conditional_probs_only/*.npz"
        save_path = f"{output_dir}/conditional_probs_only/kl_divergence.npy"
    else:
        raise ValueError("Please specify either unconditional_only or conditional_only")

    # get files
    import glob
    log_probs_original_path = glob.glob(log_probs_original_path)[0]
    log_probs_new_path = glob.glob(log_probs_new_path)[0]

    # load NPZ files
    data_original = np.load(log_probs_original_path)
    data_new = np.load(log_probs_new_path)

    # select the log_p arrays
    log_probs_original = data_original['log_p']
    log_probs_new = data_new['log_p']

    # print inputs in log probabilities
    print(f"Original log probabilities: {log_probs_original}")
    print(f"New log probabilities: {log_probs_new}")

    kl_divergence = np.sum(np.sum(np.exp(log_probs_new) * (log_probs_new - log_probs_original), axis=0), axis=1)

    print(f"KL divergence: {kl_divergence}")

    # save the KL divergence array in the output directory as npz
    np.savez(save_path, kl_divergence=kl_divergence, S=data_new['S'])

    return


