import os

from transformers import BertTokenizer, RobertaTokenizer
import torch
from dataloader import DiffusionLoader
import numpy as np
import diffusion_word_freq
import math
from tqdm import tqdm

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='lm1b', splits=['train'])[0]

word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)

for data in tqdm(train_data):
    for iid in train_data['input_ids']:
        word_freq[iid] += 1

if not os.path.exists('./word_freq'):
    os.mkdir('word_freq')

torch.save(word_freq, f'./word_freq/bert-base-uncased_lm1b.pt')



