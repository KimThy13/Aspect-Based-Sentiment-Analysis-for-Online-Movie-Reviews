import re
from sklearn.metrics import f1_score, accuracy_score
import evaluate

def evaluate_component_outputs(predictions, references):
    """
    Đánh giá component extraction dạng sinh câu.
    predictions: List[str] - Các câu được sinh
    references: List[str] - Các câu thật (ground truth)
    """
    rouge = evaluate.load("rouge")

    # Tính ROUGE
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    # Tính Exact Match
    exact_match_acc = sum([p.strip() == r.strip() for p, r in zip(predictions, references)]) / len(predictions)

    # In kết quả
    print("\n==== Component Sentence Extraction Evaluation ====")
    for k, v in rouge_scores.items():
        print(f"{k}: {v:.4f}")
    print(f"Exact Match: {exact_match_acc:.4f}")

    return {**rouge_scores, "exact_match": exact_match_acc}

# Hàm đánh giá kết quả đầu ra mô hình ABSA
def evaluate_absa_outputs(target_texts, predicted_texts):
    # Load ROUGE metric
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=predicted_texts, references=target_texts)

    # Hàm tách từng thành phần
    def extract_parts(text):
        try:
            return {
                'aspect': re.search(r'aspect:\s*(.*?),', text).group(1).strip(),
                'aspect_term': re.search(r'aspect term:\s*(.*?),', text).group(1).strip(),
                'opinion_term': re.search(r'opinion term:\s*(.*?),', text).group(1).strip(),
                'sentiment': re.search(r'sentiment:\s*(\w+)', text).group(1).strip()
            }
        except:
            return None

    # Tách và lọc những câu lỗi không parse được
    true_parts = [extract_parts(t) for t in target_texts]
    pred_parts = [extract_parts(p) for p in predicted_texts]
    true_parts, pred_parts = zip(*[(t, p) for t, p in zip(true_parts, pred_parts) if t and p])

    # Tính F1 cho từng trường
    fields = ['aspect', 'aspect_term', 'opinion_term', 'sentiment']
    f1_scores = {
        f"{f}_f1": f1_score([t[f] for t in true_parts], [p[f] for p in pred_parts], average='micro')
        for f in fields
    }

    # Accuracy và Exact Match
    other_metrics = {
        "aspect_acc": accuracy_score([t['aspect'] for t in true_parts], [p['aspect'] for p in pred_parts]),
        "sentiment_acc": accuracy_score([t['sentiment'] for t in true_parts], [p['sentiment'] for p in pred_parts]),
        "exact_match_acc": sum(t == p for t, p in zip(true_parts, pred_parts)) / len(true_parts)
    }

    # Gộp lại tất cả kết quả
    results = {**rouge_results, **f1_scores, **other_metrics}

    # In kết quả
    print("\n==== Evaluation Results ====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results
