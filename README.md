# SHL Grammar Scoring Engine for Spoken Data

A multimodal machine learning solution for the SHL Hiring Assessment 2026 competition. The goal is to predict a continuous grammar score from 0 to 5 for spoken English recordings.

## Overview

The project combines information from both the speech signal and the language contained in the speech.

The final pipeline is:

```
Audio
  ├──> Acoustic Feature Extraction ─────────────┐
  │                                             │
  └──> Whisper Speech-to-Text                   │
          ├──> TF-IDF Features                  │
          ├──> Grammar Features                 │
          └──> Linguistic / POS Features        │
                                                ↓
                                      Feature Combination
                                                ↓
                                         Ridge Regression
                                                ↓
                                      Grammar Score (0–5)
```

## Dataset

- Training samples: 769
- Test samples: 216
- Target: Continuous grammar score in the range 0–5
- Input: Spoken English WAV recordings

## Feature Engineering

### 1. Speech Transcription

Audio is transcribed using Whisper. Audio is loaded with librosa at a sampling rate of 16 kHz before transcription.

### 2. Text Features

The transcripts are represented using:

- TF-IDF features with unigrams and bigrams
- Basic grammar-related features
- Linguistic and POS-based features extracted using spaCy

### 3. Acoustic Features

Acoustic characteristics are extracted directly from the speech signal. These features provide information that is not available from the transcript alone.

### 4. Feature Scaling and Combination

Numerical feature groups are standardized using StandardScaler. The TF-IDF, grammar, linguistic, and acoustic representations are then combined into a single feature matrix.

Dependency-based features were also experimented with but were excluded from the final pipeline because they did not improve validation performance.

## Model

The final model is Ridge Regression.

The regularization parameter was selected using 5-fold cross-validation.

- Model: Ridge Regression
- Best alpha: 1.0
- 5-fold CV RMSE: 0.7593

The final model was retrained on all 769 training samples.

## Evaluation

| Approach | RMSE |
|---|---:|
| Mean baseline | 1.3596 |
| TF-IDF + Ridge | 1.2922 |
| TF-IDF + Grammar + Ridge | 1.1330 |
| TF-IDF + Grammar + Linguistic + Ridge | 1.0799 |
| Audio-only Ridge | 0.8761 |
| Multimodal Ridge, alpha = 10, 5-fold CV | 0.7852 |
| **Multimodal Ridge, alpha = 1, 5-fold CV** | **0.7593** |

The multimodal approach improved substantially over the text-only and audio-only baselines.

## Final Training Result

The final model was trained on all 769 training samples.

**Final Training RMSE: 0.4057211172**

This metric is reported on the complete training dataset after fitting the final model.

## Prediction and Submission

The final model generates predictions for all 216 test recordings. Predictions are clipped to the valid 0–5 score range.

The submission file contains:

- filename
- label
- 216 test predictions
- No missing prediction values

The Kaggle notebook contains the complete preprocessing, feature engineering, model selection, evaluation, final training, prediction, and submission pipeline.

## Project Structure

```
SHL-Competition/
├── notebook3a2ae43118.ipynb   # Complete Kaggle notebook
└── README.md                  # Project documentation
```

## Technologies

- Python
- Pandas
- NumPy
- Librosa
- Faster-Whisper
- spaCy
- Scikit-learn
- SciPy
- TF-IDF
- Ridge Regression
- Jupyter / Kaggle Notebooks