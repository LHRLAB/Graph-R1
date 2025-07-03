import numpy as np
import re
import string
from collections import Counter
from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-roberta-large")

def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def calculate_metric_scores_rsim(gold_answers, predicted_answers):
    assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

    example_eval_results = []
    total = 0

    for gold, predicted in zip(gold_answers, predicted_answers):
        score = model.similarity([normalize_answer(gold)],[normalize_answer(predicted)])
        total += score[0][0]

    avg = total / len(gold_answers) if gold_answers else 0.0
    pooled_eval_results = {"R-Sim": avg}

    return pooled_eval_results, example_eval_results

# #For Evaluation
# answers = [
#     ["Politician"],
#     ["By going to the ball."],
#     ["Rockland County"]
# ]

# pred_answers = [
#     "Politician and a good person.",
#     "By going to ball.",
#     "New York."
# ]

def cal_rsim(gold_answers, predicted_answers):
    overall_qa_rsim_result, example_qa_rsim_results = calculate_metric_scores_rsim(
        gold_answers=gold_answers, predicted_answers=predicted_answers)
    return overall_qa_rsim_result["R-Sim"]

# overall_qa_em_result, example_qa_em_results = calculate_metric_scores_em(
#     gold_answers=answers, predicted_answers=pred_answers,
#     aggregation_fn=np.max)
# overall_qa_f1_result, example_qa_f1_results = calculate_metric_scores_f1(
#     gold_answers=answers, predicted_answers=pred_answers,
#     aggregation_fn=np.max)

# # round off to 4 decimal places for QA results
# overall_qa_em_result.update(overall_qa_f1_result)
# overall_qa_results = overall_qa_em_result
# overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
# print(f"Evaluation results for QA: {overall_qa_results}")