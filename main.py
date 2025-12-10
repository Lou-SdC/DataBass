import os
from dotenv import load_dotenv
import models.baseline as baseline_model
from extract.chorus_bass_extract import extract_chorus_bass_list
from preprocess.librosa_load import load_audio_files

if __name__ == "__main__":
    print("🎸 Welcome to DataBass! Starting the processing pipeline... 🪩")

    ## ETL ##
    resp = input("Run ETL ? [Y/n]: ").strip().lower()
    if resp == "y":
        # extract raw_data and save in data/preprocessed/chorus_bass_list.csv
        load_dotenv()
        dir = os.getenv('WORKING_DIR')
        processed_file = extract_chorus_bass_list(dir)
        print(f"✅ Chorus bass extraction complete! Preprocessed data saved in {processed_file} 🎉")

        # transform the .wav files to data frames with librosa
        print(f"✨ Starting librosa loading 🌹")
        loaded_df_file = load_audio_files(processed_file)
        print(f"✅ Librosa loading finished! 🎊")
    else:
        print("Skipping ETL.")
        working_dir = os.getenv('WORKING_DIR')
        loaded_df_file = os.path.join(working_dir, 'data', 'preprocessed', 'librosa_loaded_audio.csv')

    ## BASELINE ##
    resp = input("Run baseline processing? [Y/n]: ").strip().lower()
    if resp == "y":
        # load chorus_bass_list.csv and process each audio file to get frequency and note
        # save the results in data/baseline/notes.csv
        # evaluate the results and save the evaluation in data/baseline/evaluation.txt
        print("🎷 Starting DataBass baseline processing...")
        prediction_file = baseline_model.predict(loaded_df_file)
        # prediction_file = '/home/julien/code/gridar/DataBass/data/baseline/notes.csv'
        result = baseline_model.evaluate(prediction_file)
        print(f"🥁 Evaluation Results:\n{result}")
        print(f"🚀🕺 DataBass baseline processing complete — results saved in {prediction_file} 🎉🤘😎")
    else:
        print("Skipping baseline processing.")

    resp = input("Run advanced baseline using pyin function? [Y/n]: ").strip().lower()
    if resp == "y":
        # load chorus_bass_list.csv and process each audio file to get frequency and note using pyin
        # save the results in data/baseline/pyin_notes.csv
        # evaluate the results and save the evaluation in data/baseline/pyin_evaluation.txt
        print("🎷 Starting DataBass advanced baseline processing using pyin...")
        prediction_file = baseline_model.predict_pyin(loaded_df_file)
        # prediction_file = '/home/julien/code/gridar/DataBass/data/baseline/pyin_notes.csv'
        result = baseline_model.evaluate_pyin(prediction_file)
        print(f"🥁 Advanced Evaluation Results:\n{result}")
        print(f"🚀🕺 DataBass advanced baseline processing complete — results saved in {prediction_file} 🎉🤘😎")
    else:
        print("Skipping advanced baseline processing.")

    ## Random forest model ##
