'''
    Filename: training-local.py
    Usage: debug the training process locally
'''
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch

# Set the random seed
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)

# Config
pretrain_model = "google/bert_uncased_L-12_H-768_A-12"

# https://github.com/huggingface/transformers/issues/701
# Google repo
# batch sizes: 8, 16, 32, 64, 128
# learning rates: 3e-4, 1e-4, 5e-5, 3e-5

device = "cpu"
epochs = 1
batch_size = 1
learning_rate = 3e-5

max_seq_length = 384
doc_stride = 128
max_query_length = 256

model_output_folder = "outputs"
model_output_name = "bert-base-squad2-uncased"
squad_data_folder = "data/squad"

checkpoint_folder = "checkpoints"

# Reset environment
import shutil
from pathlib import Path

shutil.rmtree(model_output_folder, ignore_errors=True)
Path(model_output_folder).mkdir(parents=True, exist_ok=True)

import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm_notebook
from transformers import AdamW,BertConfig,BertForQuestionAnswering,BertTokenizer,get_linear_schedule_with_warmup,squad_convert_examples_to_features

from transformers.data.metrics.squad_metrics import compute_predictions_logits,squad_evaluate
from transformers.data.processors.squad import SquadResult, SquadV2Processor

def wrapper():
    model_name = pretrain_model
    processor = SquadV2Processor()
    examples = processor.get_train_examples(f"{squad_data_folder}/")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)
    model.to(device)

    features,train_dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=True,
        return_dataset="pt",
    )

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    #Start training
    import tqdm
    import torch

    t_total = len(train_dataloader) * epochs

    # Prepare optimizer and schedule 
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    print("  Num examples = ", len(train_dataset))
    print("  Total optimization steps = ", t_total)

    steps = 1
    tr_loss = 0.0
    model.zero_grad()

    for epoch in range(epochs):
        print('Epoch:{}'.format(epoch+1))
        epoch_iterator = tqdm.tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()

            iter_loss = loss.item()
            tr_loss += iter_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()  
            model.zero_grad()
            steps += 1
            
            # Log 
            if steps % 100 == 0:
                print('steps = {}, loss = {}, avg loss = {}'.format(steps,iter_loss,tr_loss/steps))
        tokenizer.save_pretrained(f"{checkpoint_folder}/{model_output_name}-epoch-{epoch}")
        model.save_pretrained(f"{checkpoint_folder}/{model_output_name}-epoch-{epoch}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': tr_loss
            }, f"{checkpoint_folder}/{model_output_name}-epoch-{epoch}/state-snapshot")

    tokenizer.save_pretrained(f"{model_output_folder}/{model_output_name}")
    model.save_pretrained(f"{model_output_folder}/{model_output_name}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': tr_loss
        }, f"{model_output_folder}/{model_output_name}/state-snapshot")

if __name__ == '__main__':
    wrapper()