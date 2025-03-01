import os
import re
import shutil

# Define the mapping of long filenames to new simplified names
mutation_mapping = {
    # 8ee2_4 mutations
    "T30P": re.compile(r'8ee2_4.*30th_bias_P.*\.pdb'),
    "L98P": re.compile(r'8ee2_4.*98th_bias_P.*\.pdb'),
    "G102S": re.compile(r'8ee2_4.*102th_bias_S.*\.pdb'),
    "G102I": re.compile(r'8ee2_4.*102th_bias_I.*\.pdb'),
    "A99F": re.compile(r'8ee2_4.*99th_bias_F.*\.pdb'),
    "T101Q": re.compile(r'8ee2_4.*101th_bias_Q.*\.pdb'),

    # 7olz mutations
    "I50P": re.compile(r'7olz.*50th_bias_P.*\.pdb'),
    "T101W": re.compile(r'7olz.*101th_bias_W.*\.pdb'),
    "T101Y": re.compile(r'7olz.*101th_bias_Y.*\.pdb'),
    "T101F": re.compile(r'7olz.*101th_bias_F.*\.pdb'),
    "L28Y": re.compile(r'7olz.*28th_bias_Y.*\.pdb'),
    "L28F": re.compile(r'7olz.*28th_bias_F.*\.pdb'),
    "N58M": re.compile(r'7olz.*58th_bias_M.*\.pdb'),
    "P99R": re.compile(r'7olz.*99th_bias_R.*\.pdb'),
    "K102L": re.compile(r'7olz.*102th_bias_L.*\.pdb'),
    "D114L": re.compile(r'7olz.*114th_bias_L.*\.pdb'),

    # 8q7s mutations
    "C48W": re.compile(r'8q7s.*48th_bias_W.*\.pdb'),
    "P108Y": re.compile(r'8q7s.*108th_bias_Y.*\.pdb'),
    "Y113K": re.compile(r'8q7s.*113th_bias_K.*\.pdb'),

    # 8q93 mutations
    "S48K": re.compile(r'8q93.*48th_bias_K.*\.pdb'),
    "S48W": re.compile(r'8q93.*48th_bias_W.*\.pdb'),
    "I50P_8q93": re.compile(r'8q93.*50th_bias_P.*\.pdb'),  # Disambiguate from 7olz I50P
    "T101W_8q93": re.compile(r'8q93.*101th_bias_W.*\.pdb'),  # Disambiguate from 7olz T101W
    "T101Y_8q93": re.compile(r'8q93.*101th_bias_Y.*\.pdb'),  # Disambiguate from 7olz T101Y
}


def rename_pdb_files(directory):
    """
    Renames PDB files according to the mutation mapping and organizes each into a separate folder
    matching its new filename.

    Args:
        directory (str): The directory containing the PDB files to rename and organize.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Get list of all PDB files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pdb')]

    # Process each file
    for file in files:
        file_path = os.path.join(directory, file)
        matched = False  # Flag to check if file matches any mutation

        # Iterate through each mutation to find a match
        for mutation, pattern in mutation_mapping.items():
            if pattern.match(file):
                matched = True

                # Extract the PDB ID from the filename (assuming PDB ID is the first part before '_')
                pdb_id = file.split('_')[0]

                # Determine the mutation name for the filename
                if '_' in mutation:
                    # Handle disambiguated mutations (e.g., T101W_8q93)
                    mutation_name = mutation.split('_')[0]
                else:
                    mutation_name = mutation

                # Create the new filename
                new_filename = f"{pdb_id}_{mutation_name}.pdb"

                # Define the mutation folder name (matching the new filename without extension)
                folder_name = os.path.splitext(new_filename)[0]
                mutation_folder = os.path.join(directory, folder_name)

                # Create the mutation folder if it doesn't exist
                if not os.path.exists(mutation_folder):
                    os.makedirs(mutation_folder)
                    print(f"Created folder: {mutation_folder}")

                # Define the new file path inside the mutation folder
                new_path = os.path.join(mutation_folder, new_filename)

                try:
                    # Copy the file to the new location with the new name
                    shutil.copy2(file_path, new_path)
                    print(f"Renamed and moved: '{file}' -> '{os.path.join(folder_name, new_filename)}'")
                except Exception as e:
                    print(f"Failed to copy '{file}' to '{new_path}'. Error: {e}")

                break  # Stop checking other mutations once a match is found

        if not matched:
            print(f"No matching mutation found for file: '{file}'")

    print("Renaming and organizing completed.")



if __name__ == "__main__":
    # Get the current directory or specify your directory
    current_dir = os.getcwd()

    # Uncomment and modify the line below if your files are in a different directory
    # current_dir = "/path/to/your/pdb/files"

    rename_pdb_files(current_dir)
    print("Renaming and organizing complete!")
