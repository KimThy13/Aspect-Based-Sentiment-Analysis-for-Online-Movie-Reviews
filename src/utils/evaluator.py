import re
from sklearn.metrics import f1_score, accuracy_score
import evaluate
import json
import logging
logger = logging.getLogger(__name__)

# # Phase 1
# def evaluate_component_outputs(predictions, references, output_file=None):
#     """
#     Evaluate component extraction outputs (sentence generation task).
#     Args:
#         predictions (List[str]): Generated sentences by model.
#         references (List[str]): Ground truth sentences.
#     Returns:
#         dict: Dictionary of ROUGE scores and exact match accuracy.
#     """
#     rouge = evaluate.load("rouge")

#     # Compute ROUGE scores between predictions and references
#     rouge_scores = rouge.compute(predictions=predictions, references=references)

#     # Calculate exact match accuracy (strict string equality)
#     exact_match_acc = sum([p.strip() == r.strip() for p, r in zip(predictions, references)]) / len(predictions)

#     # Print evaluation results
#     logger.info("==== Component Sentence Extraction Evaluation ====")
#     for k, v in rouge_scores.items():
#         logger.info(f"{k}: {v:.4f}")
#     logger.info(f"Exact Match: {exact_match_acc:.4f}")

#     results = {**rouge_scores, "exact_match": exact_match_acc}

#     if output_file:
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=4)
#         logger.info(f"Saved component evaluation results to {output_file}")

#     return results

def evaluate_component_outputs(predictions, references, output_file=None):
    """
    Evaluate component extraction outputs (sentence generation task) using set matching.
    Args:
        predictions (List[str]): Generated sentences by model.
        references (List[str]): Ground truth sentences.
        output_file (str, optional): Path to save evaluation result as JSON.
    Returns:
        dict: Dictionary of F1, precision, recall, and optionally ROUGE and exact match.
    """

    # Strip and lower all sentences for consistency
    pred_set = set([p.strip().lower() for p in predictions])
    ref_set = set([r.strip().lower() for r in references])

    true_positives = len(pred_set & ref_set)
    false_positives = len(pred_set - ref_set)
    false_negatives = len(ref_set - pred_set)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    logger.info("==== Component Sentence Extraction Evaluation ====")
    logger.info(f"Set-based Precision: {precision:.4f}")
    logger.info(f"Set-based Recall: {recall:.4f}")
    logger.info(f"Set-based F1: {f1:.4f}")

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Optional: if predictions and references have same length, also compute ROUGE and exact match
    if len(predictions) == len(references):
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        exact_match_acc = sum([p.strip() == r.strip() for p, r in zip(predictions, references)]) / len(predictions)

        logger.info("ROUGE Scores (only if lengths match):")
        for k, v in rouge_scores.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info(f"Exact Match: {exact_match_acc:.4f}")

        results.update(rouge_scores)
        results["exact_match"] = exact_match_acc

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved component evaluation results to {output_file}")

    return results






# Phase 2
def evaluate_absa_outputs(target_texts, predicted_texts):
    """
    Evaluate ABSA outputs by comparing generated text to target text.
    Uses ROUGE, F1, accuracy, and exact match metrics on extracted parts.
    Args:
        target_texts (List[str]): Ground truth formatted strings.
        predicted_texts (List[str]): Model-generated formatted strings.
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predicted_texts, references=target_texts)

    def extract_parts(text):
        """
        Parse text into dictionary of aspect, aspect_term, opinion_term, sentiment.
        Return None if parsing fails.
        """
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

    # Extract parts from target and predicted texts, filter out parse errors
    true_parts = [extract_parts(t) for t in target_texts]
    pred_parts = [extract_parts(p) for p in predicted_texts]
    true_parts, pred_parts = zip(*[(t, p) for t, p in zip(true_parts, pred_parts) if t and p])

    # Calculate F1 score for each field
    fields = ['aspect', 'aspect_term', 'opinion_term', 'sentiment']
    f1_scores = {
        f"{f}_f1": f1_score([t[f] for t in true_parts], [p[f] for p in pred_parts], average='micro')
        for f in fields
    }

    # Calculate accuracy for aspect and sentiment fields and overall exact match accuracy
    other_metrics = {
        "aspect_acc": accuracy_score([t['aspect'] for t in true_parts], [p['aspect'] for p in pred_parts]),
        "sentiment_acc": accuracy_score([t['sentiment'] for t in true_parts], [p['sentiment'] for p in pred_parts]),
        "exact_match_acc": sum(t == p for t, p in zip(true_parts, pred_parts)) / len(true_parts)
    }

    # Combine all metrics into one dictionary
    results = {**rouge_results, **f1_scores, **other_metrics}

    # Print all evaluation results
    print("\n==== Evaluation Results ====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results
