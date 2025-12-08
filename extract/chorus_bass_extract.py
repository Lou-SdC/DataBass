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


def extract_chorus_bass_list(working_dir):
    directory =  working_dir + '/raw_data/Bass monophon/Lists/Chorus'
    print(f"Loading chorus bass data from {directory}...")
    files = os.listdir(directory)
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
            'Bass monophon',
            'Audio',
            'Samples',
            f"{x}.wav"
        )
    )

    print(df.head())

    output_dir = os.path.join(working_dir, 'data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'chorus_bass_list.csv')
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    print("Extracting chorus bass list...")
    load_dotenv()
    WORKING_DIR = os.getenv('WORKING_DIR')
    extract_chorus_bass_list(WORKING_DIR)
