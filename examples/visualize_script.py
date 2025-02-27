import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt




# Define the alphabet and corresponding dictionary
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = dict(zip(alphabet, range(21)))

def softmax1(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1)[:, np.newaxis]

def softmax(x):
    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def visualize_heatmap(original=False, unconditional_only=False, conditional_only=False,  start_nano_len=0, end_nano_len=0, output_dir=None):

    if original:
        # parent_path = "../outputs/example_4_outputs_original"
        parent_path = output_dir
        nano_chain = "C"
        # end_nano_len = 110
        # start_nano_len = 293

    else:
        # parent_path = "../outputs/example_4_outputs"
        parent_path = output_dir
        nano_chain = "B"
        # start_nano_len = 117
        # end_nano_len = 143

    if unconditional_only:
        path = f"{parent_path}/unconditional_probs_only"
    elif conditional_only:
        path = f"{parent_path}/conditional_probs_only"
    else:
        raise ValueError("Please specify either unconditional_only or conditional_only")


    nano_seqs = []
    soft_probs = []
    for f in glob.glob(f"{path}/*.npz"):
        # Load the .npz file
        with np.load(f) as data:
            for key in data.files:
                print(key, data[key])

            if data.files.__contains__('log_p'):
                # Extract log_probs
                log_probs = data["log_p"][0]
                print("##################################", log_probs.shape)
                # print(log_probs.shape)
            else:
                log_probs = data["log_probs"]
                # print(log_probs.shape

            #extract and convert sequence to amino acids
            S = data['S']
            if len(S.shape) > 1:
                seq = ["".join([alphabet[i] for i in s]) for s in S]
            else:
                seq = "".join([alphabet[j] for j in S])

            # print(seq)
            # check if chain_order is present
            if data.files.__contains__('chain_order'):
                # get chain_order
                chain_order = data['chain_order'][0]
                # print(chain_order)
                # print(chain_order[1])
                if chain_order[0] == nano_chain:
                    nano_seq = seq[0][:end_nano_len]
                    print(len(nano_seq))
                    log_probs = log_probs[0][:end_nano_len]
                else:
                    nano_seq = seq[1][start_nano_len::]
                    print(len(nano_seq))
                    log_probs = log_probs[1][start_nano_len::]

            else:
                nano_seq = seq[start_nano_len::]
                print(len(nano_seq))
                log_probs = log_probs[start_nano_len::]

            # save the log_probs
            # np.save(f"{f}_log_probs.npy", log_probs)

            nano_seqs.append(nano_seq)

            # Apply softmax to the first set of log_probs
            print(log_probs.shape)
            soft_probs.append(softmax(log_probs))


    # do a simple sequence alignment and mark the differences
    # assert len(nano_seqs[0]) == len(nano_seqs[1])
    # diff_positions = []
    # for i in range(len(nano_seqs[0])):
    #     if nano_seqs[0][i] != nano_seqs[1][i]:
    #         print(f"Position {i} is different: {nano_seqs[0][i]} vs {nano_seqs[1][i]}")
    #         diff_positions.append(i)

    # Plot the softmax heatmap of probabilities for each numpy array
    for i, f in enumerate(glob.glob(f"{path}/*.npz")):
        if len(soft_probs[i][0].shape) == 1:
            soft_prob = soft_probs[i]
        else:
            soft_prob = soft_probs[i][0]
        print(soft_prob.shape[0])

        # Save range of positions
        positions = np.arange(soft_prob.shape[0])
        nano_seq = nano_seqs[i]



        y_ticks = []
        for i in range(len(nano_seq)):
            # make nano_seq[i] + position
            y_ticks.append(f"{nano_seq[i]}  {i}")

        plt.figure(figsize=(16, 20))
        sns.heatmap(soft_prob, cmap="viridis", xticklabels=alphabet, yticklabels=y_ticks, cbar_kws={'label': 'Softmax probability'})

        # Mark specific positions
        cdr_region_positions = []

        # Loop through numbers from 1 to 117
        for i in range(1, 118):
            # Check if the current number is NOT in the excluded ranges
            if (28 <= i <= 34) or (54 <= i <= 58) or (100 <= i <= 111):
                cdr_region_positions.append(i)  # Append the number to list2

        custom_positions = [28, 35, 54, 59, 100, 112]
        custom_positions = [pos - 4 for pos in custom_positions]

        # Define cdr_positions based on custom_positions
        cdr_positions = [
            (custom_positions[0], custom_positions[1], 'CDR1'),  # CDR1: 24 to 30
            (custom_positions[2], custom_positions[3], 'CDR2'),  # CDR2: 50 to 54
            (custom_positions[4], custom_positions[5], 'CDR3')  # CDR3: 96 to 107
        ]

        # for pos in diff_positions:
        # for pos in cdr_region_positions:
        # for pos in custom_positions:
        #     plt.axhline(y=pos, color='red', linestyle='--')  # Horizontal line
        #     plt.text(-0.5, pos, f'{pos}↓          ', color='red', va='center', ha='right')  # Annotate the position

        # Add horizontal lines for each CDR region
        for start, end, cdr in cdr_positions:
            plt.axhline(y=start, color='red', linestyle='--', label=f'{cdr} start')
            plt.axhline(y=end, color='red', linestyle='--', label=f'{cdr} end')
            plt.text(-0.5, start, f'{cdr} ↓        ', color='red', va='center', ha='right')
            plt.text(-0.5, end, f'{cdr} ↑        ', color='red', va='center', ha='right')

        for i in range(len(alphabet)):
            plt.axvline(i, color="gray", linestyle="--", linewidth=0.7)

        for i in range(len(positions)):
            plt.axhline(i, color="gray", linestyle="--", linewidth=0.5)

        # set the title
        plt.title(f"{f.split('/')[-1]} Softmax probabilities")

        # Set the x and y labels
        plt.ylabel("Position")
        plt.xlabel("Amino Acid")

        # Save the heatmap
        plt.savefig(f"{f}.png")
        plt.show()
        plt.close()

if __name__ == "__main__":
    visualize_heatmap(original=False, conditional=False)
