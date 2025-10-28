import os
import numpy as np
from beam import endpoint, Image, Volume, QueueDepthAutoscaler
from transformers import pipeline
from pydub import AudioSegment
import io
import base64

# Set up Hugging Face cache directory
HF_HOME_DIR = './checkpoints'
CACHE_DIR = './checkpoints/cache'
os.environ['HF_HOME'] = HF_HOME_DIR

# Define the model loading function
def load_model():
    model = pipeline(
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
        task="audio-classification",
        device="cuda:0",
        cache_dir=CACHE_DIR,
    )
    return model

# Configure autoscaling
autoscaling_config = QueueDepthAutoscaler(
    max_containers=5,
    tasks_per_container=30,
)

# Define the Beam endpoint
@endpoint(
    name="audio-spectrogram-transformer",
    cpu=2,
    memory="8Gi",
    gpu=["RTX4090"],  # or ["T4", "RTX4090"] if you want to allow multiple GPU types
    image=Image(
        python_version="python3.10",
        python_packages=[
            "transformers==4.45.2",
            "torch==2.5.0",
            "pydub==0.25.1",
            "soundfile==0.12.1",
            "numpy==1.24.0",
        ],
    ),
    volumes=[
        Volume(name="audio-checkpoints", mount_path=HF_HOME_DIR),
    ],
    on_start=load_model,
    keep_warm_seconds=300,
    autoscaler=autoscaling_config,
)
def classify_audio(context, base64_audio: str, labels: list[str]):
    model = context.on_start_value
    audio_bytes = base64.b64decode(base64_audio)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

    # Convert AudioSegment to NumPy array
    samples = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        # If stereo, convert to mono by averaging channels
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    # Run inference
    predictions = model(samples, top_k=10)
    return {"predictions": predictions}
