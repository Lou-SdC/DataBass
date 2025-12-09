import os
import librosa
import shutil
from pathlib import Path

from pydub import AudioSegment

def clear_folder(folder, confirm=True):
    """
    Completely empties a folder if it exists and isn't empty.
    Delete all files and sub folders.

    Args:
        folder: path of the folder
        confirm: if True, asks for confirmation before deleting (default: True)
    """
    if os.path.exists(folder):
        if confirm:
            response = input(f"Are you sure you want to empty '{folder}' ? (yes/no) : ").strip().lower()
            if response not in ['oui', 'o', 'yes', 'y']:
                print("Operation cancelled.")
                return

        try:
            shutil.rmtree(folder)
            print(f"Folder '{folder}' successfully emptied.")
        except Exception as e:
            print(f"ERROR while emptying folder: {e}")
            raise

def audio_split_by_note(fichier_audio, dossier_sortie="notes", confirm_clear=True):
    """
    Splits an audio file by detecting note onsets using librosa.

    Args:
        fichier_audio: path to a .wav audio file
        dossier_sortie: name of the output folder (default: "notes")
        confirm_clear: if True, asks for confirmation before clearing output folder (default: True)

    Returns:
        dict: {
            'num_notes_detected': int,
            'num_files_created': int,
            'output_folder': str,
            'success': bool
        }

    Raises:
        FileNotFoundError: if the audio file doesn't exist
        ValueError: if the file is not a valid audio format
        Exception: for other processing errors
    """
    result = {
        'num_notes_detected': 0,
        'num_files_created': 0,
        'output_folder': dossier_sortie,
        'success': False
    }

    try:
        # 1. Vérifier que le fichier existe
        if not os.path.exists(fichier_audio):
            raise FileNotFoundError(f"❌ Audio file not found: '{fichier_audio}'")

        # 2. Vérifier l'extension du fichier
        valid_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
        file_ext = Path(fichier_audio).suffix.lower()
        if file_ext not in valid_extensions:
            raise ValueError(
                f"❌ Invalid audio format: '{file_ext}'\n"
                f"   Supported formats: {', '.join(valid_extensions)}\n"
                f"   File: '{fichier_audio}'"
            )

        # 3. Vérifier que le fichier n'est pas vide
        file_size = os.path.getsize(fichier_audio)
        if file_size == 0:
            raise ValueError(f"❌ Audio file is empty (0 bytes): '{fichier_audio}'")

        print(f"✓ Loading audio file: {fichier_audio} ({file_size / (1024*1024):.2f} MB)")

        # 4. Charger l'audio avec librosa
        try:
            y, sr = librosa.load(fichier_audio, sr=None)
        except Exception as e:
            raise ValueError(
                f"❌ Failed to load audio file with librosa\n"
                f"   Error: {str(e)}\n"
                f"   File: '{fichier_audio}'\n"
                f"   This may indicate the file is corrupted or in an unsupported format."
            )

        # 5. Vérifier que l'audio n'est pas vide après chargement
        if len(y) == 0:
            raise ValueError(
                f"❌ Audio file loaded but contains no samples\n"
                f"   File: '{fichier_audio}'"
            )

        duration = librosa.get_duration(y=y, sr=sr)
        print(f"✓ Audio loaded successfully - Duration: {duration:.2f}s, Sample rate: {sr} Hz")

        # 6. Détection des onsets (début des notes)
        try:
            onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            num_onsets = len(onsets)

            if num_onsets == 0:
                raise ValueError(
                    f"❌ No note onsets detected in the audio file\n"
                    f"   File: '{fichier_audio}'\n"
                    f"   Duration: {duration:.2f}s\n"
                    f"   This may indicate the audio is silent, too quiet, or has no distinct notes."
                )

            print(f"✓ Detected {num_onsets} note(s)")
            result['num_notes_detected'] = num_onsets

        except ValueError:
            raise
        except Exception as e:
            raise Exception(
                f"❌ Error during onset detection\n"
                f"   Error: {str(e)}"
            )

        # 7. Vider le dossier de sortie s'il existe
        clear_folder(dossier_sortie, confirm=confirm_clear)

        # 8. Charger aussi avec pydub pour découper
        try:
            audio = AudioSegment.from_file(fichier_audio)
        except Exception as e:
            raise Exception(
                f"❌ Failed to load audio file with pydub\n"
                f"   Error: {str(e)}\n"
                f"   File: '{fichier_audio}'"
            )

        # 9. Créer le dossier de sortie
        try:
            os.makedirs(dossier_sortie, exist_ok=True)
            print(f"✓ Output folder created/verified: '{dossier_sortie}'")
        except Exception as e:
            raise Exception(
                f"❌ Failed to create output folder\n"
                f"   Folder: '{dossier_sortie}'\n"
                f"   Error: {str(e)}"
            )

        # 10. Découper chaque segment
        files_created = 0
        for i in range(len(onsets)):
            try:
                start_ms = int(onsets[i] * 1000)  # en millisecondes
                end_ms = int(onsets[i+1] * 1000) if i+1 < len(onsets) else len(audio)

                segment = audio[start_ms:end_ms]

                # Vérifier que le segment n'est pas vide
                if len(segment) == 0:
                    print(f"⚠ Warning: Segment {i+1} is empty (0ms), skipping...")
                    continue

                nom_fichier = os.path.join(dossier_sortie, f"note_{i+1:03d}.wav")
                segment.export(nom_fichier, format="wav")
                files_created += 1
                print(f"  ✓ Note {i+1:03d} saved: {nom_fichier} ({len(segment)}ms)")

            except Exception as e:
                print(f"  ❌ Error saving note {i+1}: {str(e)}")
                continue

        result['num_files_created'] = files_created

        if files_created != num_onsets:
            print(f"⚠ Warning: {num_onsets} notes detected but only {files_created} files created")
        else:
            print(f"✓ All {files_created} note(s) successfully saved")

        result['success'] = True
        print("✓ Audio splitting completed successfully.")
        return result

    except FileNotFoundError as e:
        print(str(e))
        return result
    except ValueError as e:
        print(str(e))
        return result
    except Exception as e:
        print(f"❌ Unexpected error during audio splitting\n   Error: {str(e)}")
        return result
