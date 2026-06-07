"""
============================================================
 Speech Emotion Recognition using Wav2Vec2
 Model  : superb/wav2vec2-base-superb-er (HuggingFace)
 Type   : Transformer (Transfer Learning)
 Task   : Audio Emotion Classification
 Author : Akshay Sharma
============================================================
"""

# ── 1. Install Dependencies ──────────────────────────────
# Run this once in terminal or Colab:
# pip install transformers torch librosa

# ── 2. Imports ───────────────────────────────────────────
import os
import pickle
from transformers import pipeline

# Force progress bars to show
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"


# ── 3. Load Pretrained Wav2Vec2 Model ────────────────────
def load_model():
    print("Loading model...")
    model_id = "superb/wav2vec2-base-superb-er"
    classifier = pipeline("audio-classification", model=model_id)
    print("Model Loaded Successfully!")
    return classifier


# ── 4. Predict Emotion from Audio File ───────────────────
def predict_emotion(classifier, audio_path):
    """
    Pass an audio file path (.wav format recommended)
    Returns predicted emotion with confidence scores
    """
    print(f"\nAnalyzing: {audio_path}")
    results = classifier(audio_path)

    print("\n── Emotion Prediction Results ──")
    for r in results:
        label = r['label']
        score = round(r['score'] * 100, 2)
        print(f"  {label:10s} → {score}%")

    top = results[0]
    print(f"\n✅ Predicted Emotion : {top['label']}")
    print(f"   Confidence        : {round(top['score'] * 100, 2)}%")
    return top


# ── 5. Save Model Locally ────────────────────────────────
def save_model(classifier, save_path="./my_emotion_model"):
    print(f"\nSaving model to {save_path}...")
    classifier.model.save_pretrained(save_path)
    classifier.feature_extractor.save_pretrained(save_path)
    print("Model saved successfully!")


# ── 6. Load Saved Model ──────────────────────────────────
def load_saved_model(save_path="./my_emotion_model"):
    print(f"Loading saved model from {save_path}...")
    classifier = pipeline("audio-classification", model=save_path)
    print("Saved model loaded successfully!")
    return classifier


# ── 7. Main ──────────────────────────────────────────────
if __name__ == "__main__":

    # Load model
    classifier = load_model()

    # Save model locally
    save_model(classifier)

    # ── Test with an audio file ──
    # Replace "test_audio.wav" with your actual audio file path
    audio_path = "test_audio.wav"

    if os.path.exists(audio_path):
        predict_emotion(classifier, audio_path)
    else:
        print(f"\n⚠️  No audio file found at '{audio_path}'")
        print("   Add a .wav file and update the audio_path variable.")

    # ── Emotion Labels Reference ──
    print("\n── Emotion Labels ──────────────────")
    print("  neu = Neutral")
    print("  hap = Happy")
    print("  sad = Sad")
    print("  ang = Angry")
