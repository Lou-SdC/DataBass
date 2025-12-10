from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from preprocess import preprocess
from models import load_model

import librosa

import numpy as np
import io

#load the model

app = FastAPI()
app.state.model = load_model()

 # Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_audio')
async def receive_audio(audio: UploadFile=File(...)):
    #receive the audio file
    contents = await audio.read()

    """
    on hold for now :
    """
    #read the audio file with librosa
    y, sr = librosa.load(io.BytesIO(contents), sr=None)

    # preprocess the audio (filtering)
    y_filtered = preprocess(y, sr)

    # make the prediction
    prediction = app.state.model.predict(y_filtered)

    # return the prediction as a response
    return Response(content=str(prediction), media_type="text/plain")
