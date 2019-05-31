import torch
import csv
import os
import random
import sys
import logging

import numpy as np

from DataRow import DataRow
from DataFeatures import DataFeatures
from DataParser import get_features

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

def read_eval_data(file):
    with open(file, "r") as f:
      reader = csv.reader(f, delimiter="\t")
      lines = []
      for line in reader:
        lines.append(line)
      
      train_data = []
    for (i, line) in enumerate(lines):
        if i == 0:  # for header
            continue
        guid = "%s-%s" % ("dev", i)
        text_a = line[1]
        label = line[0]
        row = DataRow(guid=guid, text_a=text_a, label=label)
        train_data.append(row)
    return train_data

batch_size = 1
epochs = 8
gradient_accumulation_steps = 1
seed = 100
learning_rate = 5e-6
warmup_proportion = 0.2
max_sequence_length = 32
data_dir = 'bert/dataset/data/'
cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1))
output_dir = 'bert/output'
label_list = ["Required","MinCharacters","Dropdown","Data","Type"]
num_labels = len(label_list)


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# torch.cuda.set_device(1)
device = torch.device("cpu")
n_gpu = torch.cuda.device_count()

# logger.warn(n_gpu)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = BertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=True)


eval_data = read_eval_data(os.path.join(data_dir, "test.tsv"))
eval_features = get_features(eval_data, label_list, max_sequence_length, tokenizer)
logger.warn("***** Running evaluation *****")
logger.warn("  Num examples = %d", len(eval_data))
logger.warn("  Batch size = %d", batch_size)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

model.eval()
eval_loss = 0
nb_eval_steps = 0
preds = []
global_step = 0
loss = 0

for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)

    # create eval loss and other metric required by the task
    loss_fct = CrossEntropyLoss()
    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    
    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(
            preds[0], logits.detach().cpu().numpy(), axis=0)

eval_loss = eval_loss / nb_eval_steps
preds = preds[0]
print(preds)

preds = np.argmax(preds, axis=1)

logger.warn(" predictions = %s", preds)

assert len(preds) == len(all_label_ids.numpy())
result = {"accuracy": (preds == all_label_ids.numpy()).mean()}
# loss = tr_loss/global_step

result['eval_loss'] = eval_loss
result['global_step'] = global_step
result['loss'] = loss

output_eval_file = os.path.join(output_dir, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.warn("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.warn("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))