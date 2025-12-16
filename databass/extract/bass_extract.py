# it is required to have a WORKING_DIR variable pointing to the raw_data directory

# each file is an xml file with a list of audiofile
#    <audiofile>
#       <fileID>B11-28100-3311-00625</fileID>
#       <instrument>B</instrument>
#       <instrumentsetting>1</instrumentsetting>
#       <playstyle>1</playstyle>
#       <midinr>28</midinr>
#       <string>1</string>
#       <fret>00</fret>
#       <fxgroup>3</fxgroup>
#       <fxtype>31</fxtype>
#       <fxsetting>1</fxsetting>
#       <filenr>00625</filenr>
#    </audiofile>


import os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils import bass_notes
from utils import guitar_notes

# exemple directory for chorus effect :
# working_dir + '/raw_data/Bass monophon/Lists/Chorus'
def extract_guitar_list(working_dir):
    raw_data_guitar = os.path.join(working_dir, 'raw_data', 'Gitarr monophon', 'Lists')
    # list the effect directories
    effect_dirs = os.listdir(raw_data_guitar)
    output_files = []
    for effect in effect_dirs:
        effect_path = os.path.join(raw_data_guitar, effect)
        print(f"Processing effect: {effect}")
        output_files.append(extract_effect_guitar_list(effect_path, working_dir))
    # combine all effect files into one
    combined_df = pd.DataFrame()
    for effect_file in output_files:
        print(f"Combining file: {effect_file}")
        df = pd.read_csv(effect_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_output_file = os.path.join(
        working_dir,
        'data',
        'preprocessed',
        'guitar_list.csv'
    )

    combined_df.to_csv(combined_output_file, index=False)
    print(f"Combined guitar list saved to {combined_output_file}")
    print("Guitar extraction complete.")
    # remove individual effect files
    for effect in effect_dirs:
        effect_name = effect
        effect_file = os.path.join(
            working_dir,
            'data',
            'preprocessed',
            f'{effect_name}_list.csv'
        )
        os.remove(effect_file)
    return combined_output_file

def extract_effect_guitar_list(directory, working_dir):
    files = os.listdir(directory)
    effect_name = os.path.basename(directory)
    data = []

    for file in files:
        tree = ET.parse(os.path.join(directory, file))
        root = tree.getroot()
        for audiofile in root.findall('audiofile'):
            fileID = audiofile.find('fileID').text
            instrument = audiofile.find('instrument').text
            instrumentsetting = audiofile.find('instrumentsetting').text
            playstyle = audiofile.find('playstyle').text
            midinr = audiofile.find('midinr').text
            string = audiofile.find('string').text
            fret = audiofile.find('fret').text
            fxgroup = audiofile.find('fxgroup').text
            fxtype = audiofile.find('fxtype').text
            fxsetting = audiofile.find('fxsetting').text
            filenr = audiofile.find('filenr').text

            # save in a dataframe
            data.append({
                'fileID': fileID,
                'instrument': instrument,
                'instrumentsetting': instrumentsetting,
                'playstyle': playstyle,
                'midinr': midinr,
                'string': string,
                'fret': fret,
                'fxgroup': fxgroup,
                'fxtype': fxtype,
                'fxsetting': fxsetting,
                'filenr': filenr
            })

    df = pd.DataFrame(data)

    df['note_name'] = df.apply(
        lambda row: guitar_notes.get_note_name(row['string'], row['fret']),
        axis=1
    )

    df['file_path'] = df['fileID'].apply(lambda x:
        os.path.join(
            directory.replace('Lists', 'Samples'),
            f"{x}.wav"
        )
    )
    output_dir = os.path.join(working_dir, 'data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{effect_name}_list.csv')
    df.to_csv(output_file, index=False)
    print(f"Extracted guitar list for effect {effect_name} saved to {output_file}")
    return output_file


def extract_effect_bass_list(directory, working_dir):
    files = os.listdir(directory)
    effect_name = os.path.basename(directory)
    data = []

    for file in files:
        tree = ET.parse(os.path.join(directory, file))
        root = tree.getroot()
        for audiofile in root.findall('audiofile'):
            fileID = audiofile.find('fileID').text
            instrument = audiofile.find('instrument').text
            instrumentsetting = audiofile.find('instrumentsetting').text
            playstyle = audiofile.find('playstyle').text
            midinr = audiofile.find('midinr').text
            string = audiofile.find('string').text
            fret = audiofile.find('fret').text
            fxgroup = audiofile.find('fxgroup').text
            fxtype = audiofile.find('fxtype').text
            fxsetting = audiofile.find('fxsetting').text
            filenr = audiofile.find('filenr').text

            # save in a dataframe
            data.append({
                'fileID': fileID,
                'instrument': instrument,
                'instrumentsetting': instrumentsetting,
                'playstyle': playstyle,
                'midinr': midinr,
                'string': string,
                'fret': fret,
                'fxgroup': fxgroup,
                'fxtype': fxtype,
                'fxsetting': fxsetting,
                'filenr': filenr
            })

    df = pd.DataFrame(data)

    df['note_name'] = df.apply(
        lambda row: bass_notes.get_note_name(row['string'], row['fret']),
        axis=1
    )

    df['file_path'] = df['fileID'].apply(lambda x:
        os.path.join(
            directory.replace('Lists', 'Samples'),
            f"{x}.wav"
        )
    )
    output_dir = os.path.join(working_dir, 'data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{effect_name}_list.csv')
    df.to_csv(output_file, index=False)
    print(f"Extracted bass list for effect {effect_name} saved to {output_file}")
    return output_file

def extract_bass_list(working_dir):
    raw_data_bass = os.path.join(working_dir, 'raw_data', 'Bass monophon', 'Lists')
    effect_dirs = os.listdir(raw_data_bass)
    output_files = []
    for effect in effect_dirs:
        effect_path = os.path.join(raw_data_bass, effect)
        print(f"Processing effect: {effect}")
        output_files.append(extract_effect_bass_list(effect_path, working_dir))
    # combine all effect files into one
    combined_df = pd.DataFrame()
    for effect_file in output_files:
        print(f"Combining file: {effect_file}")
        df = pd.read_csv(effect_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    combined_output_file = os.path.join(
        working_dir,
        'data',
        'preprocessed',
        'bass_list.csv'
    )
    combined_df.to_csv(combined_output_file, index=False)
    print(f"Combined bass list saved to {combined_output_file}")
    print("Bass extraction complete.")
    # remove individual effect files
    for effect in effect_dirs:
        effect_name = effect
        effect_file = os.path.join(
            working_dir,
            'data',
            'preprocessed',
            f'{effect_name}_list.csv'
        )
        os.remove(effect_file)
    return combined_output_file

def extract_combined_bass_guitar_list(working_dir):
    """
    Extracts both guitar and bass lists and merges them into a single CSV.
    Returns path to the combined CSV.
    """
    # First, generate per-instrument combined CSVs using existing flows
    guitar_csv = extract_guitar_list(working_dir)
    bass_csv = extract_bass_list(working_dir)

    # Read and merge
    guitar_df = pd.read_csv(guitar_csv)
    bass_df = pd.read_csv(bass_csv)

    combined_df = pd.concat([guitar_df, bass_df], ignore_index=True)

    combined_output_file = os.path.join(
        working_dir,
        'data',
        'preprocessed',
        'instruments_list.csv'
    )
    combined_df.to_csv(combined_output_file, index=False)
    print(f"Combined instruments list saved to {combined_output_file}")
    return combined_output_file


if __name__ == "__main__":
    print("Extracting guitar and bass lists...")
    load_dotenv()
    working_dir = os.getenv('WORKING_DIR')
    if not working_dir:
        raise RuntimeError("WORKING_DIR environment variable is not set.")
    combined_csv = extract_combined_bass_guitar_list(working_dir)
    # Optionally trigger spectrogram extraction after list generation
    print(f"Done. Combined CSV: {combined_csv}")
