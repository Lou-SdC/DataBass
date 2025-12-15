import os
from dotenv import load_dotenv
import models.baseline as baseline_model
import models.pyin as pyin
import kagglehub

from extract.bass_extract import extract_bass_list
from extract.spectrograms_extract import extract_spectrograms

if __name__ == "__main__":
    print("🎸 Welcome to DataBass! Starting the processing pipeline... 🪩")
    working_dir = os.getenv('WORKING_DIR')

    ## ETL ##
    resp = input("Run ETL ? [Y/n]: ").strip().lower()
    if resp == "y":
        # extract raw_data and save in data/preprocessed/bass_list.csv
        load_dotenv()
        processed_file = extract_bass_list(working_dir)
        print(f"✅ Bass extraction complete! Preprocessed data saved in {processed_file} 🎉")
    else:
        print("Skipping ETL.")
        processed_file = os.path.join(working_dir, 'data', 'preprocessed', 'bass_list.csv')

    ## BASELINE ##
    resp = input("Run baseline processing? [Y/n]: ").strip().lower()
    if resp == "y":
        # load bass_list.csv and process each audio file to get frequency and note
        # save the results in data/baseline/notes.csv
        # evaluate the results and save the evaluation in data/baseline/evaluation.txt
        print("🎷 Starting DataBass baseline processing...")
        prediction_file = baseline_model.predict(processed_file)
        result = baseline_model.evaluate(prediction_file)
        print(f"🥁 Evaluation Results:\n{result}")
        print(f"🚀🕺 DataBass baseline processing complete — results saved in {prediction_file} 🎉🤘😎")
    else:
        print("Skipping baseline processing.")

    resp = input("Run advanced baseline using pyin function? [Y/n]: ").strip().lower()
    if resp == "y":
        # load bass_list.csv and process each audio file to get frequency and note using pyin
        # save the results in data/baseline/pyin_notes.csv
        # evaluate the results and save the evaluation in data/baseline/pyin_evaluation.txt
        print("🎷 Starting DataBass advanced baseline processing using pyin...")
        prediction_file = pyin.predict(processed_file)
        # prediction_file = '/home/julien/code/gridar/DataBass/data/baseline/pyin_notes.csv'
        result = pyin.evaluate(prediction_file)
        print(f"🥁 Advanced Evaluation Results:\n{result}")
        print(f"🚀🕺 DataBass advanced baseline processing complete — results saved in {prediction_file} 🎉🤘😎")
    else:
        print("Skipping advanced baseline processing.")


    resp = input("Run spectrogram extraction? [Y/n]: ").strip().lower()
    if resp == "y":
        # extract spectrograms
        print("Extracting spectrogram list...")
        extract_spectrograms()
        print("✅ Spectrogram extraction complete! 🎉")
