'''
    Filename: index.py
    Usage: Provide the API for the web UI
'''
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from hashlib import md5
import torch
from utils.model import question_answer

MEMORY = ""

def system_init():
    if not os.path.exists("library"):
        os.mkdir("library")

    # Load docs to memory
    for book_name in os.listdir("library"):
        with open(f"library/{book_name}", "r") as f:
            [content] = f.readlines()
            global MEMORY
            MEMORY += content
    
# Initialize Flask
app = Flask(__name__)
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
    answer = question_answer(q, MEMORY)
    return jsonify(answer=answer)


@app.route("/api/documents", methods=["POST"]) # Document API
def upload():
    content = request.json.get("content")
    if not content:
        return jsonify(error="Content is required")
    filename = f"{md5(content.encode('utf8')).hexdigest()}.txt"
    path = f"library/{filename}"
    with open(path, "w") as f:
        f.write(f"{content}")
    global MEMORY
    MEMORY += content
    print("INFO: Memory is updated")
    return jsonify(filename=filename)


if __name__ == "__main__":
    system_init()
    app.run(port=9000)
