from music21 import stream, note, meter, tempo

# --- Création d'une partition ---
score = stream.Score()

# --- Création d'une partie (instrument, ici générique) ---
part = stream.Part()
part.id = 'BassLine'  # optionnel, utile pour le nommage

# --- Ajout d’un tempo et d’une mesure ---
part.append(tempo.MetronomeMark(number=100))
part.append(meter.TimeSignature('4/4'))

# --- Ajout de notes (exemple simple C - E - G - C) ---
notes = [
    note.Note('C3', quarterLength=1),
    note.Note('E3', quarterLength=1),
    note.Note('G3', quarterLength=1),
    note.Note('C4', quarterLength=2),
]

for n in notes:
    part.append(n)

# --- Ajout de la partie à la partition ---
score.append(part)

# --- Export en MusicXML ---
score.write('musicxml', fp='ma_partition.xml')

print("Fichier 'ma_partition.xml' exporté avec succès !")
