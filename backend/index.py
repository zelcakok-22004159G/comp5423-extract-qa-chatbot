'''
    Filename: index.py
    Usage: Provide the API for the web UI
'''
from transformers import BertTokenizer
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import os
from hashlib import md5

# Prepare the model
model_name = "zelcakok/bert-base-squad2-uncased"
qa_instance = pipeline("question-answering",
                       model=model_name, tokenizer=model_name)

# Load docs to memory
isExist = os.path.exists("library")
if not isExist:
    os.mkdir("library")

memory = ["I am Chatbot who can answer questions based on the database. Please upload some documents."]
for book_name in os.listdir("library"):
    with open(f"library/{book_name}", "r") as f:
        content = f.readlines()
        memory += [line.replace("\n", "") for line in content]

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
    result = qa_instance(question=q, context=" ".join(memory))
    model_response = result["answer"]
    if result["score"] <= 0.05:
        result["answer"] = "I don't know"
    return jsonify(answer=result["answer"], metadata={"score": result["score"], "modelResponse": model_response})


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
