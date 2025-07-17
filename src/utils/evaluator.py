import re
import json
import nltk
import logging
import evaluate
from sklearn.metrics import f1_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')
logger = logging.getLogger(__name__)

# ===== Phase 1: Component Extraction =====
def evaluate_component_outputs(predictions, references, output_file=None):
    pred_set = set()
    ref_set = set()

    for p in predictions:
        pred_set.update([s.strip().lower() for s in p.split(";") if s.strip()])
    for r in references:
        ref_set.update([s.strip().lower() for s in r.split(";") if s.strip()])

    true_positives = len(pred_set & ref_set)
    false_positives = len(pred_set - ref_set)
    false_negatives = len(ref_set - pred_set)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    if len(predictions) == len(references):
        exact_match_acc = sum(p.strip() == r.strip() for p, r in zip(predictions, references)) / len(predictions)
        results["exact_match"] = exact_match_acc

        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=predictions, references=references)

        logger.info("ROUGE Scores (only if lengths match):")
        for k, v in rouge_scores.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info(f"Exact Match: {exact_match_acc:.4f}")

        results.update(rouge_scores)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved component evaluation results to {output_file}")

    return results

def evaluate_component_outputs_by_sentence(predictions, references, output_file=None):
    assert len(predictions) == len(references), "Predictions and references must be same length"

    total_prec, total_rec, total_f1 = 0, 0, 0
    total_bleu = 0
    smooth_fn = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        pred_sents = [s.strip() for s in pred.split(";") if s.strip()]
        ref_sents = [s.strip() for s in ref.split(";") if s.strip()]

        tp = sum([p in ref_sents for p in pred_sents])
        precision = tp / len(pred_sents) if pred_sents else 0
        recall = tp / len(ref_sents) if ref_sents else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0

        total_prec += precision
        total_rec += recall
        total_f1 += f1

        # BLEU per sentence (soft similarity)
        sample_bleu = 0
        for p in pred_sents:
            bleu = max(
                [sentence_bleu([nltk.word_tokenize(r)], nltk.word_tokenize(p), smoothing_function=smooth_fn) for r in ref_sents],
                default=0
            )
            sample_bleu += bleu
        if pred_sents:
            total_bleu += sample_bleu / len(pred_sents)

    n = len(predictions)
    results = {
        "mean_precision": total_prec / n,
        "mean_recall": total_rec / n,
        "mean_f1": total_f1 / n,
        "mean_bleu": total_bleu / n
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved sentence-level evaluation results to {output_file}")

    return results


# ===== Phase 2: ABSA Evaluation =====
def evaluate_absa_outputs(target_texts, predicted_texts, output_file=None):
    def extract_parts(text):
        try:
            return {
                'aspect': re.search(r'aspect:\s*(.*?),', text).group(1).strip(),
                'aspect_term': re.search(r'aspect term:\s*(.*?),', text).group(1).strip(),
                'opinion_term': re.search(r'opinion term:\s*(.*?),', text).group(1).strip(),
                'sentiment': re.search(r'sentiment:\s*(\w+)', text).group(1).strip()
            }
        except Exception as e:
            logger.warning(f"Failed to parse text: {text} - {e}")
            return None

    true_parts = [extract_parts(t) for t in target_texts]
    pred_parts = [extract_parts(p) for p in predicted_texts]
    filtered = [(t, p) for t, p in zip(true_parts, pred_parts) if t and p]

    if not filtered:
        logger.warning("No valid ABSA predictions to evaluate.")
        return {}

    true_parts, pred_parts = zip(*filtered)

    fields = ['aspect', 'aspect_term', 'opinion_term', 'sentiment']
    f1_scores = {
        f"{f}_f1": f1_score([t[f] for t in true_parts], [p[f] for p in pred_parts], average='micro')
        for f in fields
    }

    results = {**f1_scores}

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved ABSA evaluation results to: {output_file}")

    return results
    
# ===== Phase 2 (Bổ sung): ABSA Evaluation Per-class =====
def evaluate_absa_detailed(target_texts, predicted_texts, output_file=None):
    def extract_parts(text):
        try:
            return {
                'aspect': re.search(r'aspect:\s*(.*?),', text).group(1).strip(),
                'aspect_term': re.search(r'aspect term:\s*(.*?),', text).group(1).strip(),
                'opinion_term': re.search(r'opinion term:\s*(.*?),', text).group(1).strip(),
                'sentiment': re.search(r'sentiment:\s*(\w+)', text).group(1).strip()
            }
        except Exception as e:
            logger.warning(f"Failed to parse text: {text} - {e}")
            return None

    true_parts = [extract_parts(t) for t in target_texts]
    pred_parts = [extract_parts(p) for p in predicted_texts]

    filtered = [(t, p) for t, p in zip(true_parts, pred_parts) if t and p]
    if not filtered:
        logger.warning("No valid ABSA predictions to evaluate.")
        return {}

    true_parts, pred_parts = zip(*filtered)

    from sklearn.metrics import classification_report

    y_true_aspect = [t["aspect"] for t in true_parts]
    y_pred_aspect = [p["aspect"] for p in pred_parts]
    aspect_report = classification_report(y_true_aspect, y_pred_aspect, output_dict=True, zero_division=0)

    y_true_sentiment = [t["sentiment"] for t in true_parts]
    y_pred_sentiment = [p["sentiment"] for p in pred_parts]
    sentiment_report = classification_report(y_true_sentiment, y_pred_sentiment, output_dict=True, zero_division=0)

    results = {
        "aspect_report": aspect_report,
        "sentiment_report": sentiment_report
    }

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved detailed ABSA evaluation results to {output_file}")

    return results

