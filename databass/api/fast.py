from pathlib import Path
from typing import Any, List, Dict, Optional
import os
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from pydantic import BaseModel
import tempfile
import librosa
import json
import music21 as m21
import xml.etree.ElementTree as ET

import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from models import conv2D, rand_forest
from preprocess import audio_split
from pipeline import MelodyReconstructor
from preprocess.spectrograms import generate_mel_spectrogram


app = FastAPI(
    title="DataBass Model API",
    version="1.0.0"
)

_model: Any | None = None
_label_encoder: Any | None = None


class PredictionResponse(BaseModel):
    """Response model for single note prediction"""
    predicted_note: str


class SplitAudioPredictResponse(BaseModel):
    """Response model for split audio with predictions"""
    success: bool
    num_notes_detected: int
    num_notes_predicted: int
    notes_with_timings: List[Dict[str, Any]]
    melody_sequence: List[str]
    total_duration: float
    error: Optional[str] = None


async def run_full_pipeline(
    file: UploadFile = File(...),
    model_type: str = "conv2d") :
    """
    Execute the complete melody reconstruction pipeline

    This function performs the FULL pipeline in sequence:
    1. **Audio Splitting**: Detect onsets and split into individual notes
    2. **Note Prediction**: Predict each note using the specified model
    3. **Melody Reconstruction**: Assemble all predictions with timing information
    4. **Results Export**: Generate and return a xml and a midi files with all results

    The complete pipeline ensures:
    - Timing accuracy: Each note has precise onset time and duration
    - Model consistency: Uses the same model for all predictions

    Args:
        file: Audio file (WAV)
        model_type: Model to use ('conv2d' or 'randforest')

    Returns:
        XML and midi export
    """

    # ============= STEP 1: VALIDATE INPUT =============
    if not file.filename:
        raise ValueError("No filename provided")

    if model_type not in ['conv2d', 'randforest']:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'conv2d' or 'randforest'")

    print(f"\n{'='*70}")
    print(f"🎵 FULL PIPELINE EXECUTION")
    print(f"{'='*70}")
    print(f"File: {file.filename}")
    print(f"Model: {model_type.upper()}")

    # ============= STEP 2: READ AUDIO =============
    contents = await file.read()
    if not contents:
        raise ValueError("Uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(contents)
        tmp_audio_path = tmp_file.name

    # Create the MelodyReconstructor
    reconstructor = MelodyReconstructor(model_type=model_type)

    with tempfile.TemporaryDirectory() as temp_notes_dir:
        # ============= STEP 3: SPLIT AUDIO =============
        print(f"\n[STEP 1/3] Splitting audio...")
        split_success = reconstructor.split_audio(
            audio_file=tmp_audio_path,
            notes_folder=temp_notes_dir,
            confirm_clear=False
        )

        if not split_success:
            raise ValueError("Audio splitting failed")

        split_info = reconstructor.results['split']
        print(f"✓ Split complete: {split_info['num_notes_detected']} notes detected")

        # ============= STEP 4: PREDICT NOTES =============
        print(f"\n[STEP 2/3] Predicting notes with {model_type.upper()}...")
        predict_success = reconstructor.predict_notes()

        if not predict_success:
            raise ValueError("Note prediction failed")

        predictions = reconstructor.results['predictions']
        successful_predictions = [p for p in predictions if p['error'] is None]
        print(f"✓ Predictions complete: {len(successful_predictions)}/{len(predictions)} successful")

        # ============= STEP 5: RECONSTRUCT MELODY =============
        print(f"\n[STEP 3/3] Reconstructing melody...")
        reconstruction = reconstructor.reconstruct_melody()

        if not reconstruction.get('success'):
            raise ValueError("Melody reconstruction failed")

        melody_seq = reconstruction['melody_sequence']
        print(f"✓ Reconstruction complete: {len(melody_seq)} notes in sequence")

        # ============= STEP 6: GENERATE CSV =============
        # print(f"\n[STEP 4/4] Generating CSV results...")
        print(f"\n[STEP 4/4] Generating XML results...")

        # 1. Créer une partition
        score = m21.stream.Score()
        bpm = split_info['tempo']
        seconds_per_beat = 60.0 / bpm  # Durée d'une noire en secondes

        # Ajouter le tempo
        metronome = m21.tempo.MetronomeMark(number=bpm)
        score.append(metronome)

        # Créer une partie
        part = m21.stream.Part(instrumentName="Piano")
        score.append(part)

        # 2. Définir une résolution rythmique (ex: 16e de note)
        resolution = 16  # 16e de note

        # 3. Ajouter les notes avec arrondi des durées
        for note_data in melody_seq:
            # Convertir les durées en quarts de note
            duration = note_data["duration"] / seconds_per_beat
            offset = note_data["start_time"] / seconds_per_beat

            # Arrondir à la résolution choisie
            rounded_duration = round(duration * resolution) / resolution
            rounded_offset = round(offset * resolution) / resolution

            # Éviter les durées nulles ou négatives
            if rounded_duration <= 0:
                continue

            # Créer la note
            n = m21.note.Note(note_data["note"])
            n.quarterLength = rounded_duration
            n.offset = rounded_offset

            # Ajouter la note à la partie
            part.append(n)

        # 4. Sauvegarder le XML et le MIDI dans des fichiers temporaires
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp_xml_file, \
             tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_midi_file:

            # Sauvegarder le XML
            score.write('musicxml', fp=tmp_xml_file.name)
            xml_path = tmp_xml_file.name

            # Sauvegarder le MIDI
            score.write('midi', fp=tmp_midi_file.name)
            midi_path = tmp_midi_file.name

        # 5. Lire les contenus des fichiers
        with open(xml_path, 'r') as f:
            xml_results = f.read()

        with open(midi_path, 'rb') as f:
            midi_results = f.read()

        # Afficher la racine du XML
        print(f"✓ XML loaded as 'xml_results' and midi loaded as 'midi_results'")

    if os.path.exists(tmp_audio_path):
        os.remove(tmp_audio_path)
        os.remove(xml_path)
        os.remove(midi_path)

    # ============= STEP 7: COMPILE FINAL RESPONSE =============

    print(f"\n{'='*70}")
    print(f"✓ PIPELINE COMPLETE")
    print(f"{'='*70}\n")

    return xml_results, midi_results


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DataBass conv2D model API", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}") from exc
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
        tmp_file.write(contents)
        tmp_file.flush()
        try:
            signal, sr = librosa.load(tmp_file.name, sr=None)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load audio: {exc}") from exc

    global _model, _label_encoder
    if _model is None:
        _model, _label_encoder = conv2D.load_model()

    print(f'signal shape: {signal.shape}, sr: {sr}')

    predicted_note = conv2D.predict(
        signal,
        sr,
        _model,
        _label_encoder
    )

    return PredictionResponse(
        predicted_note=predicted_note,
    )

@app.post("/split_audio_predict", response_model=SplitAudioPredictResponse)
async def split_audio_predict(
    file: UploadFile = File(...),
    model_type: str = "conv2d"
) -> SplitAudioPredictResponse:
    """
    Endpoint: Split audio into notes and predict each note

    This endpoint performs 3 main steps:
    1. **Audio Splitting**: Detects note onsets and splits the audio into individual notes
    2. **Preprocessing**: Each note is preprocessed according to the model requirements
    3. **Prediction**: Each preprocessed note is sent to the model for classification

    Args:
        file: Audio file (WAV, MP3, etc.)
        model_type: Model to use ('conv2d' or 'randforest')

    Returns:
        SplitAudioPredictResponse with notes, timings, and predictions
    """

    try:
        # ============= STEP 1: VALIDATE INPUT =============
        if not file.filename:
            raise ValueError("No filename provided")

        if model_type not in ['conv2d', 'randforest']:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'conv2d' or 'randforest'")

        print(f"\n🔄 Processing file: {file.filename} with {model_type.upper()} model")

        # ============= STEP 2: READ AND LOAD AUDIO =============
        # Read the uploaded file bytes
        contents = await file.read()

        if not contents:
            raise ValueError("Uploaded file is empty")

        # Write to temporary file (required by librosa)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_audio_path = tmp_file.name

        try:
            # Load audio with librosa
            signal, sr = librosa.load(tmp_audio_path, sr=None)
            print(f"✓ Audio loaded: {len(signal)} samples at {sr} Hz")

            # ============= STEP 3: SPLIT AUDIO INTO NOTES =============
            # Create temporary working directory for split notes
            with tempfile.TemporaryDirectory() as temp_notes_dir:
                split_result = audio_split.audio_split_by_note(
                    tmp_audio_path,
                    dossier_sortie=temp_notes_dir,
                    confirm_clear=False
                )

                if not split_result['success']:
                    raise ValueError(f"Audio splitting failed: {split_result.get('error', 'Unknown error')}")

                num_notes = split_result['num_notes_detected']
                print(f"✓ Audio split into {num_notes} notes")

                # ============= STEP 4: LOAD MODEL =============
                global _model, _label_encoder
                if model_type == 'conv2d':
                    if _model is None:
                        _model, _label_encoder = conv2D.load_model()
                    model = _model
                    label_encoder = _label_encoder
                else:
                    # For randforest, load on each request (stateless)
                    model = rand_forest.load_model()
                    label_encoder = None

                print(f"✓ {model_type.upper()} model loaded")

                # ============= STEP 5: PREDICT EACH NOTE =============
                predictions = []
                notes_with_timings = []

                for i in range(1, num_notes + 1):
                    note_file = os.path.join(temp_notes_dir, f"note_{i:03d}.wav")

                    try:
                        # Load the individual note
                        note_signal, note_sr = librosa.load(note_file, sr=None)

                        # Predict based on model type
                        if model_type == 'conv2d':
                            predicted_note = conv2D.predict(note_signal, note_sr, model, label_encoder)
                        else:
                            predicted_note = rand_forest.predict(note_file, model, sr=note_sr)

                        # Get timing information
                        onset_time = split_result['onset_times'][i-1]
                        duration = split_result['note_lengths'][i-1]

                        predictions.append(predicted_note)
                        notes_with_timings.append({
                            'note_index': i,
                            'predicted_note': predicted_note,
                            'onset_time_s': float(onset_time),
                            'duration_s': float(duration),
                            'end_time_s': float(onset_time + duration)
                        })

                        print(f"  [{i:2d}/{num_notes}] {predicted_note} @ {onset_time:.2f}s (dur: {duration:.3f}s)")

                    except Exception as e:
                        print(f"  [{i:2d}/{num_notes}] ❌ Error predicting: {str(e)}")
                        continue

                print(f'    Successfully predicted notes: {len(predictions)}/{num_notes}')
                print(f'    Unsuccessfully predicted notes: {num_notes - len(predictions)}')
                print(f'    Success rate: {len(predictions)}/{num_notes} ({(len(predictions)/num_notes)*100:.2f}%)')

        finally:
            # Clean up temporary audio file
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)

        # ============= STEP 6: COMPILE RESULTS =============
        total_duration = sum(split_result['note_lengths'])

        response = SplitAudioPredictResponse(
            success=True,
            num_notes_detected=num_notes,
            num_notes_predicted=len(notes_with_timings),
            notes_with_timings=notes_with_timings,
            melody_sequence=predictions,
            total_duration=float(total_duration),
            error=None
        )

        print(f"\n✓ Prediction complete: {len(predictions)}/{num_notes} notes successfully predicted")
        return response

    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        return SplitAudioPredictResponse(
            success=False,
            num_notes_detected=0,
            num_notes_predicted=0,
            notes_with_timings=[],
            melody_sequence=[],
            total_duration=0.0,
            error=str(e)
        )
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return SplitAudioPredictResponse(
            success=False,
            num_notes_detected=0,
            num_notes_predicted=0,
            notes_with_timings=[],
            melody_sequence=[],
            total_duration=0.0,
            error=f"Internal server error: {str(e)}"
        )


@app.post("/full_pipeline_xml")
async def get_xml(file: UploadFile = File(...), model_type: str = "conv2d"):
    """
    Endpoint pour obtenir le fichier XML de la mélodie reconstruite.
    """
    xml_results, _ = await run_full_pipeline(file, model_type)
    return Response(content=xml_results, media_type='application/xml')

@app.post("/full_pipeline_midi")
async def get_midi(file: UploadFile = File(...), model_type: str = "conv2d"):
    """
    Endpoint pour obtenir le fichier MIDI de la mélodie reconstruite.
    """
    _, midi_results = await run_full_pipeline(file, model_type)
    return Response(content=midi_results, media_type='audio/midi', headers={"Content-Disposition": "attachment; filename=melody.mid"})
