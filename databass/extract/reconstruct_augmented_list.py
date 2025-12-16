"""
Reconstruct instruments_list_augmented.csv from augmented audio folders.

This script parses through the augmented audio folders (dimmed, fade_in_out,
normalized, shifted) and matches each file back to its original metadata
from instruments_list.csv.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv


def extract_original_fileID(filename, prefix):
    """
    Extract the original fileID from an augmented filename.

    Args:
        filename: e.g., 'dimmed_B11-28100-3311-00625.wav'
        prefix: e.g., 'dimmed_'

    Returns:
        fileID: e.g., 'B11-28100-3311-00625'
    """
    if filename.startswith(prefix):
        # Remove prefix and .wav extension
        return filename[len(prefix):].replace('.wav', '')
    return None


def reconstruct_augmented_list(working_dir):
    """
    Reconstruct the augmented instruments list by parsing augmented audio folders.
    """
    # Mapping of folder names to their prefixes used in filenames
    augmentation_folders = {
        'dimmed': 'dimmed_',
        'fade_in_out': 'fadeinout_',
        'normalized': 'norm_',
        'shifted': 'shifted_'
    }

    # Load original instruments list
    original_csv_path = os.path.join(working_dir, 'data', 'preprocessed', 'instruments_list.csv')

    if not os.path.exists(original_csv_path):
        raise FileNotFoundError(f"Original instruments list not found at {original_csv_path}")

    original_df = pd.read_csv(original_csv_path)
    print(f"Loaded {len(original_df)} rows from original instruments list")

    # Create a dictionary for fast lookup by fileID
    original_by_fileID = original_df.set_index('fileID').to_dict('index')

    # Start with the original data
    augmented_df = original_df.copy()
    augmented_rows = []

    # Process each augmentation folder
    for folder_name, prefix in augmentation_folders.items():
        folder_path = os.path.join(working_dir, 'data', folder_name)

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue

        print(f"\nProcessing folder: {folder_name}")

        # Get all .wav files in the folder
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"Found {len(wav_files)} .wav files")

        for wav_file in wav_files:
            # Extract original fileID
            original_fileID = extract_original_fileID(wav_file, prefix)

            if original_fileID is None:
                print(f"Warning: Could not extract fileID from {wav_file}")
                continue

            # Look up original metadata
            if original_fileID not in original_by_fileID:
                print(f"Warning: fileID {original_fileID} not found in original list")
                continue

            # Create new row with augmented file path
            original_row = original_by_fileID[original_fileID]
            new_row = original_row.copy()
            new_row['file_path'] = os.path.join(folder_path, wav_file)

            augmented_rows.append(new_row)

    print(f"\nTotal augmented rows created: {len(augmented_rows)}")

    # Combine original and augmented data
    if augmented_rows:
        augmented_additions_df = pd.DataFrame(augmented_rows)
        augmented_df = pd.concat([augmented_df, augmented_additions_df], ignore_index=True)

    # Save to CSV
    output_path = os.path.join(working_dir, 'data', 'preprocessed', 'instruments_list_augmented.csv')
    augmented_df.to_csv(output_path, index=False)

    print(f"\nSaved augmented list to {output_path}")
    print(f"Total rows: {len(augmented_df)} (original: {len(original_df)}, augmented: {len(augmented_rows)})")

    return output_path


if __name__ == "__main__":
    load_dotenv()
    working_dir = os.getenv('WORKING_DIR')

    if not working_dir:
        raise RuntimeError("WORKING_DIR environment variable is not set")

    reconstruct_augmented_list(working_dir)
    print("\nReconstruction complete!")
