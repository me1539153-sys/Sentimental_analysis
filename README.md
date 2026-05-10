# Human Emotion Recognition
A modern, deep-learning-based web application that identifies human emotions from speech audio. This project leverages the Wav2Vec2 transformer architecture to achieve high-accuracy results without the need for manual feature extraction.

 *Features*
Modern AI Core: Uses superb/wav2vec2-base-superb-er fine-tuned on professional emotional datasets.

Instant Analysis: Upload any .mp3, .wav, or .ogg file and get results in seconds.

Interactive UI: Built with Streamlit for a clean, browser-based user experience.

Confidence Metrics: Displays a breakdown of all detected emotions with probability scores.

*Tech Stack*
Language: Python 3.10+

Frontend: Streamlit

AI Engine: PyTorch & Hugging Face Transformers

Audio Processing: Librosa

How it Works
The application follows a 3-step pipeline:

Audio Processing: The uploaded file is resampled to 16kHz to match the model's requirements.

Transformer Inference: The Wav2Vec2 model extracts deep acoustic features directly from the raw waveform.

Classification: The final layer maps these features to specific emotions: Happy, Sad, Angry, Fear, Disgust, or Neutral.
