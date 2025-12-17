"""
Audio Melody Reconstruction Pipeline
=====================================

Ce module orchestre le pipeline complet :
1. Chargement d'un fichier audio long
2. Découpage en notes individuelles avec timestamps
3. Prédiction de chaque note avec un modèle (Conv2D ou RandomForest)
4. Reconstruction de la séquence complète avec timings

Auteur: DataBass Team
Date: 2025
"""

import os
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Literal
import librosa
import sys
parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))
from databass.preprocess.audio_split import audio_split_by_note
from databass.models import conv2D, rand_forest


class MelodyReconstructor:
    """
    Pipeline complet pour la reconstruction de mélodie à partir d'un fichier audio.

    Attributs:
        model_type (str): Type de modèle ('conv2d' ou 'randforest')
        model: Modèle chargé
        label_encoder: Encodeur des labels pour décoder les prédictions
        audio_file (str): Chemin du fichier audio à traiter
        notes_folder (str): Dossier contenant les notes découpées
        results (dict): Résultats du pipeline
    """

    def __init__(self, model_type: Literal['conv2d', 'randforest'] = 'conv2d'):
        """
        Initialise le reconstructeur avec le modèle spécifié.

        Args:
            model_type (str): 'conv2d' ou 'randforest'

        Raises:
            ValueError: Si le type de modèle n'est pas reconnu
        """
        if model_type not in ['conv2d', 'randforest']:
            raise ValueError(f"❌ Model type '{model_type}' not recognized. Use 'conv2d' or 'randforest'")

        self.model_type = model_type
        print(f"🔧 Initializing MelodyReconstructor with {model_type.upper()} model...")

        # Charger le modèle
        try:
            if model_type == 'conv2d':
                self.model, self.label_encoder = conv2D.load_model()
                print("✓ Conv2D model loaded successfully")
            else:
                self.model = rand_forest.load_model()
                self.label_encoder = None  # RandomForest n'a pas besoin de label encoder
                print("✓ RandomForest model loaded successfully")
        except Exception as e:
            raise Exception(f"❌ Failed to load {model_type} model: {str(e)}")

        self.audio_file = None
        self.notes_folder = None
        self.results = {}

    def split_audio(self, audio_file: str, notes_folder: str = "notes", confirm_clear: bool = False) -> bool:
        """
        Étape 1: Découpe le fichier audio en notes individuelles.

        Args:
            audio_file (str): Chemin du fichier audio
            notes_folder (str): Dossier de sortie pour les notes
            confirm_clear (bool): Demander confirmation avant de vider le dossier

        Returns:
            bool: True si succès, False sinon
        """
        print(f"\n{'='*70}")
        print(f"ÉTAPE 1: DÉCOUPAGE AUDIO")
        print(f"{'='*70}")
        print(f"📁 Audio file: {audio_file}")
        print(f"📂 Output folder: {notes_folder}")

        # Appeler la fonction de découpage
        split_result = audio_split_by_note(audio_file, notes_folder, confirm_clear=confirm_clear)

        if not split_result['success']:
            print(f"❌ Audio splitting failed!")
            return False

        # Stocker les résultats du découpage
        self.audio_file = audio_file
        self.notes_folder = notes_folder
        self.results['split'] = split_result

        # Résumé du découpage
        print(f"\n✓ SPLIT SUMMARY:")
        print(f"  - Notes detected: {split_result['num_notes_detected']}")
        print(f"  - Files created: {split_result['num_files_created']}")
        print(f"  - Onset times: {len(split_result['onset_times'])} timestamps")
        print(f"  - Note lengths: {len(split_result['note_lengths'])} durations")
        if split_result['tempo']:
            print(f"  - Estimated tempo: {split_result['tempo']:.2f} BPM")

        return True

    def predict_notes(self, sr: int = 22050) -> bool:
        """
        Étape 2: Prédit la note pour chaque fichier découpe.

        Args:
            sr (int): Sample rate pour RandomForest (Conv2D charge automatiquement)

        Returns:
            bool: True si succès, False sinon
        """
        print(f"\n{'='*70}")
        print(f"ÉTAPE 2: PRÉDICTION DES NOTES")
        print(f"{'='*70}")
        print(f"🎵 Model type: {self.model_type.upper()}")

        if not self.audio_file or not self.notes_folder:
            print("❌ Audio not split yet. Call split_audio() first!")
            return False

        predictions = []
        split_result = self.results['split']
        num_notes = split_result['num_files_created']

        print(f"🔮 Predicting {num_notes} note(s)...\n")

        for i in range(1, num_notes + 1):
            note_file = os.path.join(self.notes_folder, f"note_{i:03d}.wav")

            # Vérifier que le fichier existe
            if not os.path.exists(note_file):
                print(f"  ⚠ Warning: File not found {note_file}, skipping...")
                predictions.append({
                    'note_index': i,
                    'file': f"note_{i:03d}.wav",
                    'predicted_note': None,
                    'error': 'File not found'
                })
                continue

            try:
                # Prédire avec le modèle approprié
                if self.model_type == 'conv2d':
                    # Conv2D: charge le signal et utilise le modèle
                    signal, sr_loaded = librosa.load(note_file, sr=None)
                    predicted_note = conv2D.predict(signal, sr_loaded, self.model, self.label_encoder)
                else:
                    # RandomForest: utilise la fonction predict
                    predicted_note = rand_forest.predict(note_file, self.model, sr=sr)

                onset_time = split_result['onset_times'][i-1]
                note_length = split_result['note_lengths'][i-1]

                prediction = {
                    'note_index': i,
                    'file': f"note_{i:03d}.wav",
                    'predicted_note': predicted_note,
                    'onset_time': onset_time,
                    'duration': note_length,
                    'error': None
                }
                predictions.append(prediction)

                # Affichage formaté
                print(f"  [{i:3d}/{num_notes}] 🎵 {predicted_note:>5s} | "
                      f"Time: {onset_time:6.2f}s | Duration: {note_length:6.3f}s | "
                      f"File: {os.path.basename(note_file)}")

            except Exception as e:
                print(f"  [{i:3d}/{num_notes}] ❌ Error: {str(e)}")
                predictions.append({
                    'note_index': i,
                    'file': f"note_{i:03d}.wav",
                    'predicted_note': None,
                    'error': str(e)
                })

        # Stocker les prédictions
        self.results['predictions'] = predictions

        # Statistiques
        successful_predictions = [p for p in predictions if p['error'] is None]
        print(f"\n✓ PREDICTION SUMMARY:")
        print(f"  - Successful: {len(successful_predictions)}/{num_notes}")
        print(f"  - Failed: {num_notes - len(successful_predictions)}")

        return True

    def reconstruct_melody(self) -> Dict:
        """
        Étape 3: Reconstruit la séquence de notes avec timings.

        Returns:
            dict: Séquence de notes avec tous les métadonnées
        """
        print(f"\n{'='*70}")
        print(f"ÉTAPE 3: RECONSTRUCTION DE LA MÉLODIE")
        print(f"{'='*70}")

        if 'predictions' not in self.results:
            print("❌ No predictions available. Call predict_notes() first!")
            return {}

        predictions = self.results['predictions']
        successful = [p for p in predictions if p['error'] is None]

        print(f"📝 Assembling {len(successful)} notes into sequence...\n")

        # Créer la séquence de notes
        melody_sequence = []
        cumulative_time = 0

        for pred in successful:
            note_entry = {
                'index': pred['note_index'],
                'note': pred['predicted_note'],
                'start_time': pred['onset_time'],
                'duration': pred['duration'],
                'end_time': pred['onset_time'] + pred['duration'],
                'source_file': pred['file']
            }
            melody_sequence.append(note_entry)
            print(f"  [{note_entry['index']:3d}] {note_entry['note']:>5s} | "
                  f"{note_entry['start_time']:6.2f}s → {note_entry['end_time']:6.2f}s "
                  f"({note_entry['duration']:6.3f}s)")
            cumulative_time = note_entry['end_time']

        # Résumé
        result = {
            'success': True,
            'total_duration': cumulative_time,
            'num_notes': len(melody_sequence),
            'melody_sequence': melody_sequence,
            'notes_list': [n['note'] for n in melody_sequence],
            'timings': [(n['start_time'], n['duration']) for n in melody_sequence]
        }

        self.results['reconstruction'] = result

        print(f"\n✓ RECONSTRUCTION SUMMARY:")
        print(f"  - Total notes: {result['num_notes']}")
        print(f"  - Total duration: {result['total_duration']:.2f}s")
        print(f"  - Notes sequence: {' → '.join(result['notes_list'])}")

        return result

    def save_results_to_csv(self, output_csv: str = "melody_results.csv") -> bool:
        """
        Sauvegarde les résultats dans un fichier CSV pour visualisation.

        Args:
            output_csv (str): Chemin du fichier CSV de sortie

        Returns:
            bool: True si succès, False sinon
        """
        print(f"\n{'='*70}")
        print(f"SAUVEGARDE DES RÉSULTATS")
        print(f"{'='*70}")

        if 'reconstruction' not in self.results:
            print("❌ No reconstruction data to save. Call reconstruct_melody() first!")
            return False

        melody = self.results['reconstruction']['melody_sequence']

        try:
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'note_index', 'note', 'start_time_s', 'duration_s',
                    'end_time_s', 'source_file'
                ])
                writer.writeheader()

                for note_entry in melody:
                    writer.writerow({
                        'note_index': note_entry['index'],
                        'note': note_entry['note'],
                        'start_time_s': f"{note_entry['start_time']:.3f}",
                        'duration_s': f"{note_entry['duration']:.3f}",
                        'end_time_s': f"{note_entry['end_time']:.3f}",
                        'source_file': note_entry['source_file']
                    })

            print(f"✓ Results saved to: {output_csv}")
            return True

        except Exception as e:
            print(f"❌ Failed to save CSV: {str(e)}")
            return False

    def get_full_report(self) -> Dict:
        """
        Retourne un rapport complet avec tous les résultats.

        Returns:
            dict: Rapport détaillé du pipeline
        """
        return {
            'model_type': self.model_type,
            'audio_file': self.audio_file,
            'notes_folder': self.notes_folder,
            'split_info': self.results.get('split', {}),
            'predictions_count': len(self.results.get('predictions', [])),
            'reconstruction': self.results.get('reconstruction', {}),
            'full_results': self.results
        }

    def run_full_pipeline(self, audio_file: str, model_type: str = None,
                         output_csv: str = "melody_results.csv") -> Dict:
        """
        Exécute le pipeline complet en une seule fonction.

        Args:
            audio_file (str): Chemin du fichier audio
            model_type (str): Type de modèle à utiliser (optionnel, utilise le défaut)
            output_csv (str): Chemin du CSV de sortie

        Returns:
            dict: Résultats complets du pipeline
        """
        print(f"\n{'#'*70}")
        print(f"# DÉMARRAGE DU PIPELINE COMPLET")
        print(f"#{'='*68}#")
        print(f"Audio: {audio_file}")
        print(f"Modèle: {self.model_type.upper()}")
        print(f"{'#'*70}\n")

        # Étape 1: Découpage
        if not self.split_audio(audio_file, confirm_clear=False):
            return {'success': False, 'error': 'Audio splitting failed'}

        # Étape 2: Prédiction
        if not self.predict_notes():
            return {'success': False, 'error': 'Note prediction failed'}

        # Étape 3: Reconstruction
        reconstruction = self.reconstruct_melody()

        if not reconstruction.get('success'):
            return {'success': False, 'error': 'Melody reconstruction failed'}

        # Sauvegarde
        self.save_results_to_csv(output_csv)

        print(f"\n{'#'*70}")
        print(f"# PIPELINE COMPLÉTÉ AVEC SUCCÈS ✓")
        print(f"{'#'*70}\n")

        return self.get_full_report()


# Exemple d'utilisation
if __name__ == "__main__":
    """
    Exemple d'utilisation du pipeline complet
    """

    # Créer une instance du reconstructeur avec Conv2D
    reconstructor = MelodyReconstructor(model_type='conv2d')

    # Exécuter le pipeline complet
    # results = reconstructor.run_full_pipeline(
    #     audio_file='path/to/your/audio.wav',
    #     output_csv='melody_output.csv'
    # )

    # Ou exécuter étape par étape pour plus de contrôle:
    # reconstructor.split_audio('path/to/your/audio.wav')
    # reconstructor.predict_notes()
    # reconstructor.reconstruct_melody()
    # reconstructor.save_results_to_csv()

    # Récupérer le rapport
    # report = reconstructor.get_full_report()
    # print(report)
