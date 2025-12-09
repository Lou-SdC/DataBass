"""
Function to create the correspondancies table between the frequencies and the notes
"""


import librosa
import pandas as pd

def create_table():
    # List of the notes in a 4 strings bass guitar (E-A-D-G) on 24 cases
    #(string, case, note, frequency)
    notes_basse = []

    # Frequencies of empty strings (standard tuning: E1, A1, D2, G2)
    frequences_cordes_vide = {
        'E': 41.20,  # Mi1
        'A': 55.00,  # La1
        'D': 73.42,  # Ré2
        'G': 98.00   # Sol2
    }

    # Name of the string (lowest to highest)
    cordes = ['E', 'A', 'D', 'G']

    # Generate notes for all strings and all cases (24 cases)
    for corde in cordes:
        frequence_vide = frequences_cordes_vide[corde]
        for fret in range(0, 25):  # 0 = empty string, 24 = 24th case
            # Compute frequency for this case case (fret)
            frequence = frequence_vide * (2 ** (fret / 12))
            # Convert to note (ex: 41.20 Hz → 'E1')
            note = librosa.hz_to_note(frequence)
            note.replace('♯', '#')  # Replace ♯ with sharps if any
            notes_basse.append({
                'corde': corde,
                'case': fret,
                'note': note,
                'fréquence (Hz)': round(frequence, 2)
            })

    # Create a dataframe to display the table
    df_notes = pd.DataFrame(notes_basse)

    # Save it in a CSV file
    df_notes.to_csv('table_correspondance_notes_basse.csv', index=False)

    print("Table créée")

    return df_notes
