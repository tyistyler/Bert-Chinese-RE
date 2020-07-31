import os
import random
import logging

import torch
import numpy as np
from transformers import BertTokenizer, BertConfig

# from official_eval import official_f1
from model import RBERT
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_CLASSES = {
    'bert':(BertConfig, RBERT, BertTokenizer)
}

MODEL_PATH_MAP = {
    'bert':'bert-base-chinese'
}

ADDITIONAL_SPECIAL_TOKENS = ["<e1>","</e1>","<e2>","</e2>"]

def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]

def load_tokenizer(args):
    tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens":ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

def write_prediction(args, output_file, preds):
    """
    For official evaluation script--来自于英文关系抽取标准数据集SemEval2010_task8_scorer

    :param output_file: prediction_file_path (e.g. eval/preposed_answer.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx,relation_labels[pred]))


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m%d%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)#为 CPU 设置种子用于生成随机数，以使得结果是确定的。
    # torch.cuda.manual_seed(args.seed)# 为当前 GPU 设置种子用于生成随机数，以使得结果是确定的。
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)#为所有的 GPU 设置种子用于生成随机数，以使得结果是确定的。


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    sk_P_result_micro = precision_score(labels, preds, average='micro')
    # sk_R_result_micro = recall_score(labels, preds, average='micro')
    sk_f1_result = f1_score(labels, preds, average='macro')
    return {
        "acc": acc,
        "sk_pre": sk_P_result_micro,
        # "f1": official_f1(),
        # "sk_recall": sk_R_result_micro,
        "f1": sk_f1_result
    }






