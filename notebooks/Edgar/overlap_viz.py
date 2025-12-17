"""
Visualization script for showing overlap regions in audio splits.
Run this in the notebook as:
  exec(open('overlap_viz.py').read())
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML
import numpy as np
import librosa
from pathlib import Path
import os

# Load and prepare data
WORKING_DIR = os.environ.get('WORKING_DIR', '.')
main_path = Path(WORKING_DIR) / 'notebooks/Edgar/audio_cut1.wav'
notes_dir = Path(WORKING_DIR) / 'notebooks/Edgar/notes'

y_main, sr_main = librosa.load(main_path, sr=None)
t_main = np.arange(len(y_main)) / sr_main

# Get onsets
hoplen = 512
onset_env = librosa.onset.onset_strength(y=y_main, sr=sr_main, hop_length=hoplen)
env_times = librosa.times_like(onset_env, sr=sr_main, hop_length=hoplen)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr_main, units='frames', hop_length=hoplen)
onsets_sec = librosa.frames_to_time(onset_frames, sr=sr_main, hop_length=hoplen)

if len(onsets_sec) == 0:
    onsets_sec = np.linspace(0, t_main[-1], 5)[1:-1]

# Build figure with 3 subplots
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=(
        'Original waveform',
        'Onsets detected',
        'Split notes with overlaps (each note spans onset→next_onset)'
    )
)

# 1) Original waveform
fig.add_trace(go.Scatter(x=t_main, y=y_main, mode='lines', name='audio_cut1',
                         line=dict(color='royalblue', width=1)), row=1, col=1)

# 2) Onset strength + markers
fig.add_trace(go.Scatter(x=env_times, y=onset_env, mode='lines', name='onset_strength',
                         line=dict(color='darkorange')), row=2, col=1)
fig.add_trace(go.Scatter(x=onsets_sec, y=np.interp(onsets_sec, env_times, onset_env,
                         left=onset_env.min(), right=onset_env.max()),
                         mode='markers', name='onsets', marker=dict(color='green', size=8, symbol='x')), row=2, col=1)

# 3) Split notes with color-coded overlap regions
split_files = sorted(notes_dir.glob('*.wav'))
colors = ['rgb(99,110,250)', 'rgb(239,85,59)', 'rgb(0,204,150)', 'rgb(171,99,250)', 'rgb(255,161,90)']

# Recreate note boundaries (onset[i] to onset[i+1])
edges = np.concatenate([[0.0], onsets_sec, [t_main[-1]]])
for i in range(len(edges) - 1):
    start, end = edges[i], edges[i + 1]
    color = colors[i % len(colors)]

    # Colored span for each note region
    fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.15,
                  line=dict(color=color, width=1), row=3, col=1)

    # Load and plot actual split audio if it exists
    if i < len(split_files):
        y_split, sr_split = librosa.load(split_files[i], sr=None)
        t_split = start + np.arange(len(y_split)) / sr_split
        # Normalize and offset for visibility
        y_split_norm = (y_split / (np.max(np.abs(y_split)) + 1e-9)) * 0.4 + (i % 5) * 0.25
        fig.add_trace(go.Scatter(x=t_split, y=y_split_norm, mode='lines',
                                name=f'note_{i+1:03d}', line=dict(color=color, width=1.5)), row=3, col=1)

# Onset markers on bottom plot
fig.add_trace(go.Scatter(x=onsets_sec, y=[0]*len(onsets_sec), mode='markers',
                         name='onset positions', marker=dict(color='black', size=6, symbol='line-ns')), row=3, col=1)

fig.update_layout(
    height=1000,
    template='plotly_white',
    title='Split notes show overlaps: each note spans from its onset to the next onset (explains 2.18× duration)',
    legend=dict(orientation='v', y=0.5, x=1.02),
    margin=dict(t=80, b=60, l=60, r=150),
    hovermode='x unified'
)

fig.update_xaxes(title_text='Time (s)', row=3, col=1)
fig.update_yaxes(title_text='Amplitude', row=1, col=1)
fig.update_yaxes(title_text='Onset strength', row=2, col=1)
fig.update_yaxes(title_text='Note extracts (stacked, color=region)', row=3, col=1)

display(HTML(fig.to_html(full_html=False, include_plotlyjs='cdn')))
