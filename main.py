import os
from dotenv import load_dotenv
import models.baseline as baseline_model
from extract.chorus_bass_extract import extract_chorus_bass_list

if __name__ == "__main__":
    print("🎸 Welcome to DataBass! Starting the processing pipeline... 🪩")

    # extract raw_data and save in data/preprocessed/chorus_bass_list.csv
    load_dotenv()
    dir = os.getenv('WORKING_DIR')
    processed_file = extract_chorus_bass_list(dir)
    # processed_file = '/home/julien/code/gridar/DataBass/data/preprocessed/chorus_bass_list.csv'
    print(f"✅ Chorus bass extraction complete! Preprocessed data saved in {processed_file} 🎉")

    ## BASELINE ##
    # load chorus_bass_list.csv and process each audio file to get frequency and note
    # save the results in data/baseline/notes.csv
    # evaluate the results and save the evaluation in data/baseline/evaluation.txt

    print("🎵 Starting DataBass baseline processing... 🥁")

    prediction_file = baseline_model.predict(processed_file)
    # prediction_file = '/home/julien/code/gridar/DataBass/data/baseline/notes.csv'
    result = baseline_model.evaluate(prediction_file)

    print(f"🎵✨ Evaluation Results:\n{result}\n🎷🎚️")
    print(f"🚀🕺 DataBass baseline processing complete — results saved in {prediction_file} 🎉🤘😎")
