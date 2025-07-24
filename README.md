# Aspect-Based Sentiment Analysis for Online Movie Reviews

This repository implements a two-phase pipeline for Aspect-Based Sentiment Analysis (ABSA) on English movie reviews using T5 and BART models. The system identifies aspect-level opinions and sentiment polarities from user-generated review texts.

## Overview

The pipeline consists of two main phases:

1. **Component Sentence Extraction**
   Splits a full movie review into separate component sentences, where each sentence expresses an independent opinion.

2. **Aspect-Based Sentiment Analysis (ABSA)**
   For each component sentence, the model extracts:

   * `aspect`: the aspect category (e.g., `Movie::Plot`)
   * `aspect term`: the word or phrase referring to the aspect
   * `opinion term`: the opinionated word or phrase
   * `sentiment`: the sentiment polarity (`Positive`, `Negative`, `Neutral`)

## Project Structure

```
.
project/
│
├── data/
│   ├── raw/                             # Raw CSV data uploaded by the user
│   │   └── full_reviews.csv
│   ├── processed/                       
│   └── preprocessed/                    # Split + tokenized datasets ready for training
│
├── models/
│   ├── component_extraction/         # Trained model checkpoints for component extraction
│   └── absa/                         # Trained model checkpoints for ABSA extraction
│
├── phase1_component_extraction/
│   └── preprocess.py                    # Preprocessing for component data
├── phase2_absa_extraction/
│   └── preprocess.py                    # Preprocessing for ABSA data
│
├── pipeline/
│   └── run_pipeline.py                  # Full pipeline: raw input → final ABSA results
│
├── utils/
│   ├── clean_data.py                   # Cleaning data
│   ├── data_loader.py                   # Load preprocessed tokenized datasets
│   ├── prepare_data.py                  # Load and format raw CSV to usable datasets
│   ├── evaluator.py                     # Evaluation functions for ABSA predictions
│   ├── trainer.py                       # Generic T5 trainer for both phases
│   ├── logger.py                       # logging
│   └── predictor.py                     # Inference script
│
├── config/                              # (Optional) Hyperparameter and model config files
│   ├── component_config.json
│   ├── absa_config.json
│   └── pipeline_config.json
│
├── main.py                              # Central CLI for training/inference/pipeline
├── requirements.txt                     # Required Python packages
├── .gitignore                           # Ignore cache, models, logs, etc.
└── README.md                            # Project documentation
```

## Quick Start

### 1. Clean Raw Reviews (Optional)
```bash
python clean_reviews.py --input data/raw/raw_reviews.csv --output data/raw/full_dataset.csv
```
### 2. Prepare Data
```bash
python src/utils/prepare_data.py --input data/raw/full_dataset.csv --output data/processed/
```
### 3. Train and Evaluate Models
```bash
python main.py \
  --prepare_data \
  --train_component \
  --eval_component \
  --train_absa \
  --eval_absa
```
### 4. Run Full Pipeline on New Reviews
```bash
python main.py \
  --run_pipeline \
  --pipeline_input data/raw/full_reviews.csv \
  --pipeline_output pipeline/output.csv
```
## Model Settings

* Architectures: T5-base, BART-base
* Epochs: 20
* Batch size: 8
* Learning rate: 3e-4
* Optimizer: AdamW
* Beam search: 4
* Evaluation: Precision, Recall, F1-score, BLEU, ROUGE

## Output Format
```bash
Each ABSA prediction is returned as a single string in the following format:

aspect: Acting, aspect term: acting, opinion term: superb, sentiment: Positive
```
## Requirements

Install dependencies via:
```bash
pip install -r requirements.txt
```
### Key Libraries

* transformers
* datasets
* torch
* scikit-learn
* evaluate
* rouge\_score
* emoji
* langdetect
* nltk
* beautifulsoup4
* pandas, tqdm, seaborn, matplotlib

## Citation

This project is part of the graduation thesis by:

**Nguyễn Hải Ngọc Huyền**\\
**Tạ Hoàng Kim Thy**
University of Science, VNU-HCM
Major: Data Science, Class of 2025
