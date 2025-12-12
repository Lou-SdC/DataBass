# dictionary to map guitar string and fret to note name
note_names = {
    0: 'C', 1: 'Cx', 2: 'D', 3: 'Dx', 4: 'E', 5: 'F',
    6: 'Fx', 7: 'G', 8: 'Gx', 9: 'A', 10: 'Ax', 11: 'B'
}

def get_note_name(string, fret):
    # guitar standard tuning E2 A2 D3 G3 B3 E4 (MIDI)
    string_tuning = {
        '1': 40,  # E2
        '2': 45,  # A2
        '3': 50,  # D3
        '4': 55,  # G3
        '5': 59,  # B3
        '6': 64   # E4
    }
    midi_number = string_tuning[string] + int(fret)
    note_index = midi_number % 12
    octave = (midi_number // 12) - 1
    return f"{note_names[note_index]}{octave}"
