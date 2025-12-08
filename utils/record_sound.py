import argparse
import pathlib
import sounddevice as sd
import soundfile as sf


def record_audio(duration: float, sample_rate: int, output_path: pathlib.Path) -> pathlib.Path:
    frames = int(duration * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)
    return output_path


def playback_audio(file_path: pathlib.Path) -> None:
    data, sample_rate = sf.read(str(file_path), dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and playback a short audio sample.")
    parser.add_argument("--duration", type=float, default=3.0, help="Recording length in seconds.")
    parser.add_argument("--samplerate", type=int, default=44100, help="Audio sample rate.")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("recorded_sample.wav"),
                        help="Destination WAV file.")
    args = parser.parse_args()
    output_path = record_audio(args.duration, args.samplerate, args.output)
    playback_audio(output_path)


if __name__ == "__main__":
    main()
