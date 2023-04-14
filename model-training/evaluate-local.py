'''
    Filename: evaluate-local.py
    Usage: Debug the evaluate process
'''
# -*- coding: utf-8 -*-
import os
import shutil
from json import dumps
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer, BertForQuestionAnswering,
                          BertTokenizer, squad_convert_examples_to_features)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits, squad_evaluate)
from transformers.data.processors.squad import SquadResult, SquadV2Processor

def wrapper():
    # Config
    device = "cpu"
    batch_size = 1

    max_seq_length = 384
    doc_stride = 128
    max_query_length = 256

    model_output_folder = "checkpoints"
    model_output_name = "bert-base-squad2-uncased-epoch-0"


    model_folder = model_output_folder
    model_name = model_output_name

    processor = SquadV2Processor()
    examples = processor.get_dev_examples('data/squad/')

    tokenizer = BertTokenizer.from_pretrained(f"{model_folder}/{model_name}")

    features,eval_dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt"
    )

    # Evaluate


    test_model = BertForQuestionAnswering.from_pretrained(f"{model_folder}/{model_name}")
    test_model.to(device)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    print("  Num examples = ", len(eval_dataset))

    all_results = []

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        test_model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            feature_indices = batch[3]

            outputs = test_model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [(output[i]).detach().cpu().tolist() for output in outputs.to_tuple()]
            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    output_folder = f"evaluate-result/{model_name}"
    shutil.rmtree(output_folder, ignore_errors=True)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size=20,
        max_answer_length=30,
        do_lower_case=True,
        output_prediction_file=f"{output_folder}/predictions.json",
        output_nbest_file=f"{output_folder}/nbest_predictions.json",
        output_null_log_odds_file=f"{output_folder}/null_odds_predictions.json",
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=0.0,
        tokenizer=tokenizer,)

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)

    with open(f"{output_folder}/result.json", "w") as f:
        f.write(dumps(results, indent=4))

if __name__ == '__main__':
    wrapper()