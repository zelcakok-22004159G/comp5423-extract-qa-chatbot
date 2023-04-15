'''
    Filename: utils/model.py
    Usage: group the model related functions.
'''
from transformers import BertTokenizer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BertForQuestionAnswering
import torch

# Prepare the model
model_name = "zelcakok/bert-base-squad2-uncased"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Code example: https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626
def question_answer(question, context):
    # Tokenize the question
    input_ids = tokenizer.encode(
        question, 
        context, 
        max_length=512, 
        truncation=True,
    )
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_idx+1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)

    # Get output from model
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    # The answer is found
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    # The answer isn't found
    if answer_end < answer_start or answer.startswith("[CLS]"):
        answer = "I don't know the answer"
    return answer.capitalize()

