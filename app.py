import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import librosa
import os

# Set page tab display
st.set_page_config(
   page_title="Audio To Partition Uploader",
   page_icon= '🎸',
   layout="wide",
   initial_sidebar_state="expanded",
)

# Example local Docker container URL
# url = 'http://api:8000'
# Example localhost development URL
# url = 'http://localhost:8000'
load_dotenv()
url = os.getenv('API_URL')


# App title and description
st.header('Audio To Partition Uploader 📸')
st.markdown('''
            > This is a Le Wagon boilerplate for any data science projects that involve exchanging audio between a Python API and a simple web frontend.

            > **What's here:**

            > * [Streamlit](https://docs.streamlit.io/) on the frontend
            > * [FastAPI](https://fastapi.tiangolo.com/) on the backend
            > * [Librosa](https://librosa.org/) to process audio files
            > * Backend and frontend can be deployed with Docker
            ''')

st.markdown("---")

### Create a native Streamlit file upload input
st.markdown("### Let's find what's the partition of this : 👇")
audio_file_buffer = st.file_uploader('Upload an audio file')

if audio_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the spectrograph of the audio user uploaded
    caption = "Here's the Audio you uploaded ☝️"
    y, sr = librosa.load(audio_file_buffer, sr=None)
    st.audio(audio_file_buffer, format='audio/wav')

    # Generate and display spectrogram
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    st.pyplot(plt)

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      audio_bytes = audio_file_buffer.getvalue()

      ### Make request to  API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_audio", files={'audio': audio_bytes})

      if res.status_code == 200: #evertything went fine
        ### Display the image of the partition returned by the API
        st.image(res.content, caption="Partition returned from API ☝️")
      else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)
