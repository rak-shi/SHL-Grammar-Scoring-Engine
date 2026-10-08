# SHL Grammar Scoring Engine

An advanced speech-based machine learning system for predicting English grammar scores from 45–60 second speech recordings.

## Overview

The system predicts a continuous grammar score from 0–5 using multimodal speech features.

## Approach

The pipeline combines three feature groups:

- Wav2Vec2-large speech embeddings
- Acoustic and prosodic features
- Whisper ASR-derived linguistic features

These features are processed using scaling and PCA and passed through an ensemble of:

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Ridge Regression
- SVR

A Ridge stacking model combines the base-model predictions.

## Linguistic Features

Whisper transcripts are used to derive features such as:

- Word count
- Unique vocabulary
- Type-token ratio
- Sentence count
- Average sentence length
- Speaking rate
- Words per minute
- Filler-word ratio
- Repeated-word ratio
- Average word length
- Long-word ratio

## Results

### Cross-validation

| Version | OOF RMSE | OOF Pearson |
|---|---:|---:|
| V3 | 0.6685 | 0.8417 |
| V4 | **0.6493** | **0.8515** |

### Kaggle leaderboard

Best submitted score:

**0.5097**

Previous score:

**0.5522**

The V4 system improved the public leaderboard score by approximately 7.7%.

## Feature Architecture

Wav2Vec2 embeddings: 1024  
Acoustic features: 25  
Linguistic features: 21  

Total raw features: **1070**

After PCA and scaling:

**145 final features**

## Tech Stack

- Python
- PyTorch
- Transformers
- Wav2Vec2
- Whisper
- Librosa
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Pandas
- NumPy

## How to Run

1. Download/attach the SHL Hiring Assessment 2026 dataset in Kaggle.
2. Open `SHL_Grammar_Scoring_Engine.ipynb`.
3. Install the required packages.
4. Run the notebook sequentially.
5. The final submission is generated as:

`submission_v4.csv`

## Note

The competition dataset and audio files are not included in this repository.
