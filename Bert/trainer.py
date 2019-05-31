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

def read_train_data(file):
    with open(file, "r") as f:
      reader = csv.reader(f, delimiter="\t")
      lines = []
      for line in reader:
        lines.append(line)
      
      train_data = []
    for (i, line) in enumerate(lines):
        if i == 0:  # for header
            continue
        guid = "%s-%s" % ("train", i)
        text_a = line[1]
        label = line[0]
        row = DataRow(guid=guid, text_a=text_a, label=label)
        train_data.append(row)
    return train_data

batch_size = 4
epochs = 1
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
train_data = read_train_data(os.path.join(data_dir, "train.tsv"))
num_train_optimization_steps = int(len(train_data) / batch_size) * epochs


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# torch.cuda.set_device(1)
device = torch.device("cpu")
n_gpu = torch.cuda.device_count()

# logger.warn(n_gpu)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',cache_dir=cache_dir,num_labels=num_labels)
model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
  {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
  {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,lr=learning_rate,warmup=warmup_proportion,t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

train_features = get_features(train_data, label_list, max_sequence_length, tokenizer);

logger.warn("***** Running training *****")
logger.warn("  Num examples = %d", len(train_data))
logger.warn("  Batch size = %d", batch_size)
logger.warn("  Num steps = %d", num_train_optimization_steps)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

train_sampler = RandomSampler(dataset)

train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

model.train()

for _ in trange(int(epochs), desc="Epoch"):
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch

    # define a new function to compute loss values for both output_modes
    logits = model(input_ids, segment_ids, input_mask, labels=None)

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

    if n_gpu > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps

    loss.backward()

    tr_loss += loss.item()
    nb_tr_examples += input_ids.size(0)
    nb_tr_steps += 1
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

# Save a trained model, configuration and tokenizer
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)

