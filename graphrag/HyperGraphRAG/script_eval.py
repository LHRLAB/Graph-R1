import json
from transformers import GPT2Tokenizer
import argparse
import os

# 初始化 GPT-2 分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 分词函数
def gpt2_tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    return tokens

# 加载 Groundtruth 和预测文件
def load_data(groundtruth_file, predictions_file):
    with open(groundtruth_file, 'r') as gt_file:
        groundtruth = json.load(gt_file)
    with open(predictions_file, 'r') as pred_file:
        predictions = json.load(pred_file)
    return groundtruth, predictions

# 计算 Context Recall (C-Rec)
def compute_c_rec(groundtruth, predictions):
    recalls = []
    for gt, pred in zip(groundtruth, predictions):
        gt_tokens = set(gpt2_tokenize(gt['result']))
        pred_tokens = set(gpt2_tokenize(pred['result']))
        overlap = len(gt_tokens.intersection(pred_tokens))
        recalls.append(overlap / len(gt_tokens) if len(gt_tokens) > 0 else 0)
    return sum(recalls) / len(recalls)

# 计算 Context Entity Recall (C-ERec)
def compute_c_erec(groundtruth, predictions, entity_extractor):
    recalls = []
    for gt, pred in zip(groundtruth, predictions):
        gt_entities = set(entity_extractor(gt['result']))
        pred_entities = set(entity_extractor(pred['result']))
        overlap = len(gt_entities.intersection(pred_entities))
        recalls.append(overlap / len(gt_entities) if len(gt_entities) > 0 else 0)
    return sum(recalls) / len(recalls)

# 简单的实体抽取函数（假设实体是以大写开头的单词）
def simple_entity_extractor(text):
    return [word for word in gpt2_tokenize(text) if word.istitle()]

# 计算 Answer Relevance (A-Rel)
def compute_a_rel(groundtruth, predictions):
    relevances = []
    for gt, pred in zip(groundtruth, predictions):
        gt_tokens = set(gpt2_tokenize(gt['query']))
        pred_tokens = set(gpt2_tokenize(pred['result']))
        overlap = len(gt_tokens.intersection(pred_tokens))
        relevances.append(overlap / len(gt_tokens) if len(gt_tokens) > 0 else 0)
    return sum(relevances) / len(relevances)

# 主函数
def evaluate_metrics(groundtruth_file, predictions_file, output_score_path):
    if "overall" in groundtruth_file:
        groundtruth1, predictions1 = load_data(groundtruth_file.replace("overall","easy"), predictions_file.replace("overall","easy"))
        groundtruth2, predictions2 = load_data(groundtruth_file.replace("overall","medium"), predictions_file.replace("overall","medium"))
        groundtruth3, predictions3 = load_data(groundtruth_file.replace("overall","hard"), predictions_file.replace("overall","hard"))
        groundtruth = groundtruth1 + groundtruth2 + groundtruth3
        predictions = predictions1 + predictions2 + predictions3
    else:
        groundtruth, predictions = load_data(groundtruth_file, predictions_file)

    c_rec = compute_c_rec(groundtruth, predictions)
    c_erec = compute_c_erec(groundtruth, predictions, simple_entity_extractor)
    a_rel = compute_a_rel(groundtruth, predictions)

    print(f"Context Recall (C-Rec): {c_rec:.4f}")
    print(f"Context Entity Recall (C-ERec): {c_erec:.4f}")
    print(f"Answer Relevance (A-Rel): {a_rel:.4f}")
    
    ## 保存结果
    answer_scores = {
        "Context Recall (C-Rec)": c_rec,
        "Context Entity Recall (C-ERec)": c_erec,
        "Answer Relevance (A-Rel)": a_rel
    }
    if not os.path.exists(os.path.dirname(output_score_path)):
        os.makedirs(os.path.dirname(output_score_path))
    
    with open(output_score_path, "w") as f:
        json.dump(answer_scores, f, indent=2)
        
    print(f"All scores saved to {output_score_path}")    
    
    

parser = argparse.ArgumentParser()
parser.add_argument("--cls", type=str, default="hypertension")
parser.add_argument("--level", type=str, default="hard")
parser.add_argument("--method", type=str, default="HyperGraphRAG")
args = parser.parse_args()

groundtruth_file=f"datasets/questions/{args.cls}/{args.cls}_{args.level}.json"
predictions_file=f"output/{args.method}/{args.cls}/{args.level}/{args.cls}_{args.level}_result.json"
output_score_path=f"output/{args.method}/{args.cls}/{args.level}/batch_eval_scores.json"

# 调用评估函数
evaluate_metrics(groundtruth_file, predictions_file, output_score_path)
