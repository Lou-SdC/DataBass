import argparse
import pathlib
import numpy as np
import soundfile as sf


def resample_waveform(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return waveform
    duration = waveform.shape[0] / float(source_sr)
    target_length = max(1, int(round(duration * target_sr)))
    source_times = np.linspace(0.0, duration, waveform.shape[0], endpoint=False, dtype=np.float64)
    target_times = np.linspace(0.0, duration, target_length, endpoint=False, dtype=np.float64)
    return np.interp(target_times, source_times, waveform.astype(np.float64, copy=False)).astype(np.float32)


def normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(waveform))
    return waveform if peak == 0 else waveform / peak


def frame_waveform(waveform: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    if waveform.shape[0] < frame_size:
        waveform = np.pad(waveform, (0, frame_size - waveform.shape[0]), mode="constant")
    last_start = waveform.shape[0] - frame_size
    starts = np.arange(0, last_start + 1, hop_length, dtype=int)
    frames = [waveform[start:start + frame_size] for start in starts]
    if starts.size == 0 or starts[-1] != last_start:
        tail = waveform[last_start:last_start + frame_size]
        frames.append(tail)
    return np.stack(frames).astype(np.float32)


def preprocess_audio(input_path: pathlib.Path, output_path: pathlib.Path, target_sr: int,
                     frame_size: int, hop_length: int) -> pathlib.Path:
    waveform, source_sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = resample_waveform(waveform.astype(np.float32, copy=False), source_sr, target_sr)
    waveform = normalize_waveform(waveform)
    frames = frame_waveform(waveform, frame_size, hop_length)
    window = np.hanning(frame_size).astype(np.float32, copy=False)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path,
             sample_rate=target_sr,
             frame_size=frame_size,
             hop_length=hop_length,
             waveform=waveform.astype(np.float32, copy=False),
             magnitude_spectrogram=magnitude,
             rms=rms)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an audio recording for analysis.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the recorded WAV file.")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("processed_sample.npz"),
                        help="Destination NPZ file for processed tensors.")
    parser.add_argument("--target-samplerate", type=int, default=16000,
                        help="Sample rate to enforce before feature extraction.")
    parser.add_argument("--frame-size", type=int, default=1024, help="Frame size for spectral analysis.")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length between frames.")
    args = parser.parse_args()
    if args.frame_size <= 0 or args.hop_length <= 0:
        parser.error("frame-size and hop-length must be positive.")
    if args.hop_length > args.frame_size:
        parser.error("hop-length must not exceed frame-size.")
    preprocess_audio(args.input, args.output, args.target_samplerate, args.frame_size, args.hop_length)


if __name__ == "__main__":
    main()
