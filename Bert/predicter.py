import torch
import csv
import os
import random
import sys
import logging

import numpy as np

from Bert.DataRow import DataRow
from Bert.DataFeatures import DataFeatures
from Bert.DataParser import get_features

from io import open
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)

logger = logging.getLogger(__name__)
level = logging.getLevelName('INFO')
logger.setLevel(level)
#logger = logging.getLogger(__name__)

batch_size = 1
epochs = 8
gradient_accumulation_steps = 1
seed = 100
learning_rate = 5e-6
warmup_proportion = 0.2
max_sequence_length = 32
cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
output_dir = 'bert/output'
label_list = ["Required","MinCharacters","Dropdown","Data","Type"]
num_labels = len(label_list)
predict_data = []
device = torch.device("cpu")

def parse_data(accptCriteriaLs):
    pred_data = [];
    for line in accptCriteriaLs:
        row = DataRow(guid="", text_a=line, label="Required")
        pred_data.append(row)
    return pred_data

def predict_rules(accptCriteriaLs):
    predict_data = parse_data(accptCriteriaLs)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)

    features = get_features(predict_data, label_list, max_sequence_length, tokenizer)

    logger.warn("***** Running Prediction *****")
    logger.warn("  Num examples = %d", len(predict_data))
    logger.warn("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    predict_rows = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    sampler = SequentialSampler(predict_rows)
    dataloader = DataLoader(predict_rows, sampler=sampler, batch_size=batch_size)

    preds = [];

    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Predicting"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        logits = model(input_ids, segment_ids, input_mask, labels=None)

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    print(preds)
    preds = np.argmax(preds, axis=1)
    predicted_labels = [];
    for i in range(len(preds)): 
        print(label_list[preds[i]])
        predicted_labels.append(label_list[preds[i]]);

    logger.warn(" predictions = %s", preds)

    return predicted_labels;


# test = [
#     "Both username and password are required",
#     "password must be atleast 8 characters"
# ]
# predict_rules(test);        