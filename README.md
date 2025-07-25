# Aspect-Based Sentiment Analysis for Online Movie Reviews

This repository implements a two-phase pipeline for Aspect-Based Sentiment Analysis (ABSA) on English movie reviews using T5 and BART models. The system identifies aspect-level opinions and sentiment polarities from user-generated review texts.

## Overview

The pipeline consists of two main phases:

1. **Component Sentence Extraction**
   Splits a full movie review into separate component sentences, where each sentence expresses an independent opinion.

2. **Aspect-Based Sentiment Analysis (ABSA)**
   For each component sentence, the model extracts:

   * `aspect`: the aspect category (e.g., `Movie`)
   * `aspect term`: the word or phrase referring to the aspect (e.g., `movie, film,..`)
   * `opinion term`: the opinionated word or phrase (e.g., `bad, good,..`)
   * `sentiment`: the sentiment polarity (`Positive`, `Negative`, `Neutral`)

## Project Structure

```
.
project/
│
├── data/
│   ├── raw/                             # Raw CSV data uploaded by the user
│   │   └── absa_reviews.csv
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
├──outputs/
│   ├── component_model/              # Saved fine-tuned component model
│   ├── absa_model/                   # Saved fine-tuned ABSA model
│   ├── component_result/             # Predictions and metrics of component extraction
│   └── absa_result/                  # Predictions and metrics of ABSA extraction
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
Train and evaluate both stages:

- Stage 1: Component Extraction

- Stage 2: ABSA Prediction

```bash
python main.py \
  --prepare_data \
  --train_component \
  --eval_component \
  --train_absa \
  --eval_absa \
  --component_model_type t5 \
  --absa_model_type bart
```
📌 Option:

- `--component_model_type` and `--absa_model_type`: `t5` of `bart`
- The results and output model will be saved at `outputs/`

---
### 4. Run Full Pipeline on New Reviews

Run full pipeline from raw reviews → component → ABSA:
```bash
python main.py \
  --run_pipeline \
  --pipeline_input data/raw/full_reviews.csv \
  --pipeline_output pipeline/output.csv \
  --component_model_type bart \
  --absa_model_type t5
```

📄 `pipeline/output.csv` will contain: | full\_review | component | aspect | aspect\_term | opinion\_term | sentiment |

---

### Training Configuration (Optional)

| Flag              | Default | Description                                             |
| ----------------- | ------- | ------------------------------------------------------- |
| `--epochs`        | 20      | Number of training epochs                               |
| `--batch_size`    | 8       | Batch size per device                                   |
| `--lr`            | 3e-4    | Learning rate                                           |
| `--warmup_steps`  | 500     | Warm-up steps for learning rate scheduler               |
| `--weight_decay`  | 0.01    | Weight decay coefficient for optimizer (regularization) |
| `--save_strategy` | epoch   | When to save checkpoints (`epoch`, `steps`, etc.)       |
| `--max_length`    | 64      | Maximum generation length for predictions               |
| `--num_beams`     | 4       | Number of beams used in beam search (generation)        |
| `--output_dir`    | outputs | Directory to save models and evaluation results         |

---

## ⚙️ Model Settings

* Architectures: T5-base, BART-base
* Evaluation: Precision, Recall, F1-score, BLEU, ROUGE

---

## 📝 Output Format

### 🔹 Component Extraction (Phase 1)

Each input review is split into one or more **component sentences**.
Output format: a list of shorter sentences focusing on specific parts/aspects of the review.

**Example:**

```
Input: The acting was great but the plot was too slow.
Output: ["The acting was great ; the plot was too slow"]
```

### 🔹 ABSA Extraction (Phase 2)

Each component sentence is converted into a structured string that captures the sentiment context.

**Format:**

```
aspect: <category>, aspect term: <term>, opinion term: <term>, sentiment: <label>
```

**Example:**

```
aspect: Acting, aspect term: acting, opinion term: superb, sentiment: Positive
```

---

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
* pandas, tqdm


## 📚 Citation

This project was developed as part of the undergraduate thesis by:

**Nguyễn Hải Ngọc Huyền**

**Tạ Hoàng Kim Thy**

University of Science, Vietnam National University - Ho Chi Minh City (VNU-HCM)

Major: Data Science — Class of 2025


