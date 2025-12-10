

# Récupérer WORKING_DIR
WORKING_DIR = os.getenv("WORKING_DIR")

# Dossier de sortie pour les spectrogrammes
output_dir = WORKING_DIR + "/spectrograms"
os.makedirs(output_dir, exist_ok=True)

# Go through the dataframe
for index, row in sample_list.iterrows():

    audio_path = row['file_path']

    # check that the file exists
    if not os.path.exists(audio_path):
        print(f"⚠️ Fichier introuvable : {audio_path}")
        continue

    # create the Mel-spectrogramme
    try:
        y, sr = librosa.load(audio_path)
        mel_spec = generate_mel_spectrogram(y, sr, normalize='minmax',
                                            target_shape=(128,128))
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {audio_path}: {e}")
        continue

    # output folder for each note
    note_dir = os.path.join(output_dir, row['note_name'])
    os.makedirs(note_dir, exist_ok=True)

    # output file name (.npy)
    output_filename = f"{row['fileID']}.npy"
    output_path = os.path.join(note_dir, output_filename)

    # Save the spectrogram .npy
    np.save(output_path, mel_spec)
