# script to calculate the KL divergence between two probability distributions


# imports
import numpy as np
from scipy.stats import entropy


def softmax(x):
    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# main function
def calculate_symmetrized_kl_divergence(output_dir, unconditional_only=False, conditional_only=False, default_mode=False):
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
        save_path = f"{output_dir}/unconditional_probs_only/kl_divergence"
    elif conditional_only:
        log_probs_original_path = f"{output_dir_original}/conditional_probs_only*.npz"
        log_probs_new_path = f"{output_dir}/conditional_probs_only/*.npz"
        save_path = f"{output_dir}/conditional_probs_only/kl_divergence"
    elif default_mode:
        log_probs_original_path = f"{output_dir_original}/probs/*.npz"
        log_probs_new_path = f"{output_dir}/probs/*.npz"
        save_path = f"{output_dir}/kl_divergence"
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

    # apply softmax to convert log probabilities to probabilities
    log_probs_original = softmax(log_probs_original)
    log_probs_new = softmax(log_probs_new)

    # print inputs in log probabilities
    print(f"Original log probabilities: {log_probs_original}")
    print(f"New log probabilities: {log_probs_new}")

    # calculate the symmetrized KL divergence
    kl_divergence_Px_Qx = np.sum(np.sum(np.exp(log_probs_new) * np.log(log_probs_new / log_probs_original), axis=0), axis=1)
    kl_divergence_Qx_Px = np.sum(np.sum(np.exp(log_probs_original) * np.log(log_probs_original / log_probs_new), axis=0), axis=1)
    symmetrized_KL_divergence = 0.5*(kl_divergence_Px_Qx + kl_divergence_Qx_Px)


    print(f"KL divergence: {symmetrized_KL_divergence}")

    # save the KL divergence array in the output directory as npz
    np.savez(save_path, kl_divergence=symmetrized_KL_divergence, S=data_new['S'])

    return


def calculate_Shanon_entropy(output_dir, unconditional_only=False, conditional_only=False, default_mode=False):
    """
    Calculate the Shannon entropy of a set of log probabilities.

    Args:
        log_probs (np.ndarray): Log probabilities of the distribution.

    Returns:
        float: Shannon entropy of the distribution.
    """
    # log probabilities path
    if unconditional_only:
        log_probs_path = f"{output_dir}/unconditional_probs_only/*.npz"
        save_path = f"{output_dir}/unconditional_probs_only/shannon_entropy"
    elif conditional_only:
        log_probs_path = f"{output_dir}/conditional_probs_only/*.npz"
        save_path = f"{output_dir}/conditional_probs_only/shannon_entropy"
    elif default_mode:
        log_probs_path = f"{output_dir}/probs/*.npz"
        save_path = f"{output_dir}/shannon_entropy"
    else:
        raise ValueError("Please specify either unconditional_only or conditional_only")

    # get files
    import glob
    log_probs_path = glob.glob(log_probs_path)[0]

    # load NPZ files
    data = np.load(log_probs_path)

    # select the log_p arrays
    log_probs = data['log_p']

    # convert log probabilities to probabilities
    log_probs = softmax(log_probs)

    # calculate the Shannon entropy
    entropy = np.sum(np.sum(np.exp(log_probs) * -log_probs, axis=0), axis=1)

    print(f"Shannon entropy: {entropy}")

    # save the KL divergence array in the output directory as npz
    np.savez(save_path, shannon_entropy=entropy, S=data['S'])

    return

def calculate_CDR_score(output_dir, unconditional_only=False, conditional_only=False, default_mode=False, cdr_ranges=None, nano_start=0, nano_end=0):

    # log probabilities path
    if unconditional_only:
        data_path = f"{output_dir}/unconditional_probs_only/*.npz"
        save_path = f"{output_dir}/unconditional_probs_only/CDR_scores"
    elif conditional_only:
        data_path = f"{output_dir}/conditional_probs_only/*.npz"
        save_path = f"{output_dir}/conditional_probs_only/CDR_scores"
    elif default_mode:
        data_path = f"{output_dir}/probs/*.npz"
        save_path = f"{output_dir}/CDR_scores"
    else:
        raise ValueError("Please specify either unconditional_only or conditional_only")

    # get files
    import glob
    f = glob.glob(data_path)

    # load NPZ files
    for file in f:
        with np.load(file) as data:
            for key in data.files:
                if key == 'log_p':
                    log_probs = data[key]
                elif key == 'log_probs':
                    log_probs = data[key]

                # extract and convert sequence to amino acids
                elif key == 'S':
                    S = data[key]

    # CDR scores
    CDR_score = []

    for batch_indx in range(S.shape[0]):
        # select the log_p arrays
        batch_log_prob = log_probs[batch_indx]
        batch_S = S[batch_indx]

        # get the CDR regions based on cdr_ranges
        CDR1_S = batch_S[cdr_ranges[0][0]:cdr_ranges[0][1]]
        CDR1_log_p = batch_log_prob[cdr_ranges[0][0]:cdr_ranges[0][1]]
        CDR2_S = batch_S[cdr_ranges[1][0]:cdr_ranges[1][1]]
        CDR2_log_p = batch_log_prob[cdr_ranges[1][0]:cdr_ranges[1][1]]
        CDR3_S = batch_S[cdr_ranges[2][0]:cdr_ranges[2][1]]
        CDR3_log_p = batch_log_prob[cdr_ranges[2][0]:cdr_ranges[2][1]]

        CDR1_log_ps = []
        CDR2_log_ps = []
        CDR3_log_ps = []
        # getting the corresponding log_prob scores based on the CDR regions selected ( take the log_prob.shape[2]
        for S_indx, S_value in enumerate(CDR1_S):
            CDR_pos_log = CDR1_log_p[S_indx][S_value]
            # calculate the average of CDR regions scores
            CDR1_log_ps.append(CDR_pos_log)
        for S_indx, S_value in enumerate(CDR2_S):
            CDR_pos_log = CDR2_log_p[S_indx][S_value]
            CDR2_log_ps.append(CDR_pos_log)
        for S_indx, S_value in enumerate(CDR3_S):
            CDR_pos_log = CDR3_log_p[S_indx][S_value]
            CDR3_log_ps.append(CDR_pos_log)


        batch_cdr_score = (np.mean(CDR1_log_ps), np.mean(CDR2_log_ps), np.mean(CDR3_log_ps))
        CDR_score.append(batch_cdr_score)

    # save the list of CDR scores in the output directory as npz
    np.savez(save_path, CDR_scores=CDR_score, S=S)

    return
