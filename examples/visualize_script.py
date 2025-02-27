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

        if "kl_divergence" in f:
            visualize_kl_divergence(f)
            continue
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
        if "kl_divergence" in f:
            continue
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


def visualize_kl_divergence(npz_file, plot_type='both', top_n=5):
    """
    Visualize KL divergence values from an NPZ file.

    Parameters:
    -----------
    npz_file : str
        Path to the NPZ file containing KL divergence data
    plot_type : str, optional
        Type of plot to generate: 'bar', 'line', or 'both' (default)
    top_n : int, optional
        Number of highest values to highlight

    Returns:
    --------
    dict
        Dictionary containing the positions and values of the top N highest KL divergence values
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import os

    # Load the .npz file
    with np.load(npz_file) as data:
        # Extract data
        kl_divergence = data["kl_divergence"]
        S = data["S"]
        print(f"KL divergence shape: {kl_divergence.shape}")
        print(f"S shape: {S.shape}")

        # Reshape kl_divergence to 2D if needed
        kl_divergence = kl_divergence.reshape(kl_divergence.shape[0], -1)

        # Compute average KL divergence per position if more than one value per position
        if kl_divergence.shape[1] > 1:
            kl_vals = np.mean(kl_divergence, axis=1)
        else:
            kl_vals = kl_divergence.flatten()

        positions = np.arange(len(kl_vals))

        # Find indices of top N values
        top_indices = np.argsort(kl_vals)[-top_n:][::-1]
        top_values = kl_vals[top_indices]

        # Base filename for saving plots
        base_filename = os.path.splitext(npz_file)[0]

        # Create bar plot
        if plot_type in ['bar', 'both']:
            plt.figure(figsize=(14, 7))
            bars = plt.bar(positions, kl_vals, color='steelblue', alpha=0.7)

            # Highlight top N values
            for idx in top_indices:
                bars[idx].set_color('red')
                bars[idx].set_alpha(1.0)

            # Add value labels only for top N bars to avoid clutter
            for idx in top_indices:
                height = kl_vals[idx]
                plt.text(idx, height, f'Pos {idx}: {height:.3f}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.title(f"KL Divergence by Position - Top {top_n} Highlighted", fontsize=14)
            plt.xlabel("Position", fontsize=12)
            plt.ylabel("KL Divergence", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add legend for highlighted values
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='steelblue', alpha=0.7, label='Regular Values'),
                Patch(facecolor='red', label=f'Top {top_n} Values')
            ]
            plt.legend(handles=legend_elements, loc='upper right')

            # If there are many positions, limit the x-axis ticks
            if len(positions) > 20:
                plt.gca().xaxis.set_major_locator(MaxNLocator(20))

            plt.tight_layout()
            plt.savefig(f"{base_filename}_bar_plot.png", dpi=300)
            plt.show()
            plt.close()

        # Create line plot (better for larger sequences)
        if plot_type in ['line', 'both']:
            plt.figure(figsize=(14, 7))

            # Plot the main line
            plt.plot(positions, kl_vals, color='steelblue', linewidth=2, alpha=0.8)

            # Highlight the top N points
            plt.scatter(top_indices, top_values, color='red', s=100, zorder=5)

            # Add annotations for top points
            for i, (idx, val) in enumerate(zip(top_indices, top_values)):
                # Alternate annotation positions to avoid overlap
                vert_offset = 0.02 * max(kl_vals) * (1 if i % 2 == 0 else 1.5)
                plt.annotate(f'Pos {idx}: {val:.3f}',
                             xy=(idx, val),
                             xytext=(idx, val + vert_offset),
                             fontsize=10,
                             ha='center',
                             va='bottom',
                             fontweight='bold',
                             arrowprops=dict(arrowstyle='->', lw=1.5, color='black', alpha=0.7))

            plt.title(f"KL Divergence by Position - Top {top_n} Highlighted", fontsize=14)
            plt.xlabel("Position", fontsize=12)
            plt.ylabel("KL Divergence", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='steelblue', lw=2, label='KL Divergence'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label=f'Top {top_n} Values')
            ]
            plt.legend(handles=legend_elements, loc='upper right')

            # If there are many positions, limit the x-axis ticks
            if len(positions) > 20:
                plt.gca().xaxis.set_major_locator(MaxNLocator(20))

            plt.tight_layout()
            plt.savefig(f"{base_filename}_line_plot.png", dpi=300)
            plt.show()
            plt.close()

        # Return information about top values
        top_info = {
            'positions': top_indices.tolist(),
            'values': top_values.tolist()
        }

        print(f"\nTop {top_n} KL divergence values:")
        for i, (pos, val) in enumerate(zip(top_indices, top_values), 1):
            print(f"{i}. Position {pos}: {val:.4f}")

        return top_info

if __name__ == "__main__":
    visualize_heatmap(original=False, unconditional_only=False, conditional_only=False)
