import os
import librosa
import shutil

from pydub import AudioSegment

def clear_folder(dossier, confirm=True):
    """
    Vide complètement un dossier s'il existe et n'est pas vide.
    Supprime tous les fichiers et sous-dossiers.

    Args:
        dossier: chemin du dossier à vider
        confirm: si True, demande une confirmation avant de supprimer (défaut: True)
    """
    if os.path.exists(dossier):
        if confirm:
            response = input(f"Êtes-vous sûr de vouloir vider le dossier '{dossier}' ? (oui/non) : ").strip().lower()
            if response not in ['oui', 'o', 'yes', 'y']:
                print("Opération annulée.")
                return

        try:
            shutil.rmtree(dossier)
            print(f"Dossier '{dossier}' vidé avec succès.")
        except Exception as e:
            print(f"Erreur lors du nettoyage du dossier : {e}")

def audio_split_by_note(fichier_audio, dossier_sortie="notes"):
    try:


        # Charger l'audio avec librosa
        y, sr = librosa.load(fichier_audio, sr=None)  # sr=None pour garder la fréquence d'origine

        # Détection des onsets (début des notes)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        print(f"Nombre de notes détectées : {len(onsets)}")

        # Charger aussi avec pydub pour découper
        audio = AudioSegment.from_file(fichier_audio)

        # Créer le dossier de sortie
        os.makedirs(dossier_sortie, exist_ok=True)

        # Découper chaque segment
        for i in range(len(onsets)):
            start_ms = int(onsets[i] * 1000)  # en millisecondes
            end_ms = int(onsets[i+1] * 1000) if i+1 < len(onsets) else len(audio)

            segment = audio[start_ms:end_ms]
            nom_fichier = os.path.join(dossier_sortie, f"note_{i+1}.wav")
            segment.export(nom_fichier, format="wav")
            print(f"Note {i+1} sauvegardée : {nom_fichier}")

        print("Découpage terminé.")

    except Exception as e:
        print(f"Erreur : {e}")
