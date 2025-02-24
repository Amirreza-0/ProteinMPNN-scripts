#!/usr/bin/env python3

import os
import subprocess
import sys
import Bio.PDB.PDBParser as PDBParser
from visualize_original import visualize_heatmap


def get_fixed_positions(pdb, original=False):

    # open the pdb file
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb)

    # get chain lengths
    # Initialize a dictionary to store chain names and their lengths

    chain_lengths = {}
    # Iterate through the models, chains, and residues
    for model in structure:
        for chain in model:
            print(list(chain.get_residues()))

            # remove all hetero atoms
            for residue in list(chain.get_residues()):
                if residue.id[0] != " ":
                    chain.detach_child(residue.id)

            # Get the chain ID
            chain_id = chain.id
            # Count the number of residues in the chain
            residue_count = len(list(chain.get_residues()))

            print(f"Chain {chain_id} has {residue_count} residues.")
            # Store the chain ID and its length
            chain_lengths[chain_id] = residue_count

    # get chain names
    chain_names = list(chain_lengths.keys())

    # Get the lengths of chains A and second chain
    chain_A_len = chain_lengths["A"]
    if "B" in chain_names:
        nano_chain_len = chain_lengths["B"]
    else:
        nano_chain_len = chain_lengths["C"]

    # Define chains to design
    if original:
        # conditions for original
        chains_to_design = "A C"
        fixed_positions_A = " ".join(str(i) for i in range(1, chain_A_len))  # 1 to 293
        fixed_positions_B = []

        # Loop through numbers from 1 to 117
        for i in range(1, nano_chain_len):
            # Check if the current number is NOT in the excluded ranges
            if not ((28 <= i <= 34) or (54 <= i <= 58) or (100 <= i <= 111)):
                fixed_positions_B.append(i)  # Append the number to list2

        fixed_positions_B = [x - 3 for x in fixed_positions_B][3:]  # 1 to 117
        fixed_positions_B = " ".join(str(i) for i in fixed_positions_B)

    else:
        # conditions for long names
        chains_to_design = "A B"
        fixed_positions_A = " ".join(str(i) for i in range(1, chain_A_len))  # 1 to 143
        fixed_positions_B = " ".join(str(i) for i in range(1, nano_chain_len))  # 1 to 110

    fixed_positions = f"{fixed_positions_A}, {fixed_positions_B}"
    # remove the parantheses
    fixed_positions = fixed_positions.replace("[", "").replace("]", "")
    print(f"Fixed positions: {fixed_positions}")

    # Get the start and end of the nano chain start=len(chain_A)+1, end=len(chain_A)+len(chain_B)
    nano_start = chain_A_len + 1
    # get whole length
    nano_end = chain_A_len + nano_chain_len


    return chains_to_design, fixed_positions, nano_start, nano_end


def main(original=False, conditional=False, chains_to_design="A", fixed_positions="1,2,3,4,5,6,7,8,9,10", folder_with_pdbs="../inputs/PDB_complexes/pdbs/pdbs/"):
    # Activate the conda environment
    # Note: Activating a conda environment within a Python script is not straightforward.
    # It's recommended to activate the environment before running this script.
    # If you must activate it within the script, you can modify the PATH accordingly.
    # Example:
    # activate_command = "source activate mlfold"
    # subprocess.run(activate_command, shell=True, executable="/bin/bash", check=True)
    # However, this approach has limitations.

    # Define directories
    # folder_with_pdbs = "../inputs/PDB_complexes/pdbs/pdbs"
    folder_with_pdbs = folder_with_pdbs

    if original:
        output_dir = f"../outputs/example_4_outputs_original/{file}"
    else:
        output_dir = f"../outputs/example_4_outputs/{file}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    path_for_parsed_chains = os.path.join(output_dir, f"parsed_pdbs.jsonl")
    path_for_assigned_chains = os.path.join(output_dir, "assigned_pdbs.jsonl")
    path_for_fixed_positions = os.path.join(output_dir, "fixed_pdbs.jsonl")



    # Define the list of commands to execute
    commands = [
        [
            "python",
            "../helper_scripts/parse_multiple_chains.py",
            "--input_path", folder_with_pdbs,
            "--output_path", path_for_parsed_chains
        ],
        [
            "python",
            "../helper_scripts/assign_fixed_chains.py",
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_assigned_chains,
            "--chain_list", chains_to_design
        ],
        [
            "python",
            "../helper_scripts/make_fixed_positions_dict.py",
            "--input_path", path_for_parsed_chains,
            "--output_path", path_for_fixed_positions,
            "--chain_list", chains_to_design,
            "--position_list", fixed_positions
        ],
        [
            "python",
            "../protein_mpnn_run.py",
            "--jsonl_path", path_for_parsed_chains,
            "--chain_id_jsonl", path_for_assigned_chains,
            "--fixed_positions_jsonl", path_for_fixed_positions,
            "--out_folder", output_dir,
            "--num_seq_per_target", "1",
            "--sampling_temp", "0.1",
            "--seed", "37",
            "--batch_size", "1",
            # "--save_probs", "1",
            # "--conditional_probs_only", "1"
            # "--score_only", "1",
            "--unconditional_probs_only", f"{int(not conditional)}",
        ]
    ]

    # Execute each command sequentially
    for cmd in commands:
        try:
            print(f"Executing: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing: {' '.join(cmd)}", file=sys.stderr)
            sys.exit(e.returncode)

    print("All commands executed successfully.")
    return output_dir

if __name__ == "__main__":
    original = True
    conditional = False
    # Define the folder with PDB files
    folder_with_pdbs_folder = "../inputs/PDB_complexes/pdbs/pdbs"
    # Loop through the files in the folder
    for folder in os.listdir(folder_with_pdbs_folder):
        for file in os.listdir(os.path.join(folder_with_pdbs_folder, folder)):
            folder_with_pdbs = os.path.join(folder_with_pdbs_folder, folder)
            # Get the fixed positions
            chains_to_design, fixed_positions, nano_start, nano_end = get_fixed_positions(os.path.join(folder_with_pdbs, file), original=original)
            print(f"Chains to design: {chains_to_design}")
            print(f"Fixed positions: {fixed_positions}")
            print(f"Folder with PDBs: {folder_with_pdbs}")
            output_dir = main(original=original, conditional=conditional, chains_to_design=chains_to_design, fixed_positions=fixed_positions, folder_with_pdbs=folder_with_pdbs)
            visualize_heatmap(original=original, conditional=conditional, output_dir=output_dir, start_nano_len=nano_start, end_nano_len=nano_end)
