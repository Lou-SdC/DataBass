# dictionary to map bass string and fret to note name
note_names = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}
def get_note_name(string, fret):
    # bass standard tuning E1 A1 D2 G2
    string_tuning = {
        '1': 28,  # E1
        '2': 33,  # A1
        '3': 38,  # D2
        '4': 43   # G2
    }
    midi_number = string_tuning[string] + int(fret)
    note_index = midi_number % 12
    octave = (midi_number // 12) - 1
    return f"{note_names[note_index]}{octave}"
