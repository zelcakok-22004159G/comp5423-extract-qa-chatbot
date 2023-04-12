'''
    Filename: index.py
    Usage: Provide the API for the web UI
'''
from transformers import BertTokenizer
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import os
from hashlib import md5
import torch

# Prepare the model
model_name = "zelcakok/bert-base-squad2-uncased"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Code example: https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626
def question_answer(question, text):
    # Tokenize the question
    input_ids = tokenizer.encode(
        question, 
        text, 
        add_special_tokens=True, 
        max_length=512, 
        truncation=True,
        padding="max_length",
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
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    return answer.capitalize()



# Load docs to memory
isExist = os.path.exists("library")
if not isExist:
    os.mkdir("library")

memory = ["I am Chatbot who can answer questions based on the database. Please upload some documents."]
for book_name in os.listdir("library"):
    with open(f"library/{book_name}", "r") as f:
        content = f.readlines()
        memory += content

# Initialize Flask
UPLOAD_FOLDER = 'library'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

@app.route("/") # Default Endpoint API
def main():
    return jsonify(status="ready")


@app.route("/api/answer") # Answer API
def answer():
    args = request.args
    q = args.get("q")
    if not q:
        return jsonify(answer="")
    answer = question_answer(q, " ".join(memory))
    return jsonify(answer=answer)


@app.route("/api/documents", methods=["POST"]) # Document API
def upload():
    content = request.json.get("content")
    if not content:
        return jsonify(error="Content is required")
    content = content.replace("\n", "")
    filename = f"{md5(content.encode('utf8')).hexdigest()}.txt"
    path = f"library/{filename}"
    with open(path, "w") as f:
        f.write(f"{content}")
    memory.append(content)
    print("INFO: Memory is updated")
    return jsonify(filename=filename)


if __name__ == "__main__":
    app.run(port=9000)
