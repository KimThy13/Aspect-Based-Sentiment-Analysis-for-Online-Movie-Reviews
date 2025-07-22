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
├── main.py                         # Main controller for training, evaluation, and pipeline execution
├── clean_reviews.py               # Script to clean raw review text
├── data/
│   ├── raw/                       # Raw input CSV files
│   ├── processed/                # Output of data splitting (train/val/test)
│   └── preprocessed/             # Tokenized datasets for Phase 1 and 2
├── models/                        # Fine-tuned models
├── pipeline/                      # Output from end-to-end pipeline
├── src/
│   ├── phase1_component_extraction/
│   │   └── preprocess.py         # Preprocessing logic for Phase 1
│   ├── phase2_aspect_extraction/
│   │   └── preprocess.py         # Preprocessing logic for Phase 2
│   └── utils/
│       ├── prepare_data.py       # Train/val/test splitting logic
│       ├── data_loader.py        # Load datasets using HuggingFace DatasetDict
│       ├── trainer.py            # Training wrapper using Seq2SeqTrainer
│       ├── predictor.py          # Prediction module
│       └── evaluator.py          # Evaluation metrics
```

## Quick Start

### 1. Clean Raw Reviews (Optional)
```bash
python clean\_reviews.py --input data/raw/raw\_reviews.csv --output data/raw/full\_dataset.csv
```
### 2. Prepare Data Splits
```bash
python src/utils/prepare\_data.py --input data/raw/full\_dataset.csv --output data/processed/
```
### 3. Train and Evaluate Models
```bash
python main.py;
\--prepare\_data;
\--train\_component;
\--eval\_component;
\--train\_absa;
\--eval\_absa
```
### 4. Run Full Pipeline on New Reviews
```bash
python main.py;
\--run\_pipeline;
\--pipeline\_input data/raw/full\_reviews.csv;
\--pipeline\_output pipeline/output.csv
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

**Nguyễn Hải Ngọc Huyền**
**Tạ Hoàng Kim Thy**
University of Science, VNU-HCM
Major: Data Science, Class of 2025
