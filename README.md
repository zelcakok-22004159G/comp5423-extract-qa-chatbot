# COMP5423 Question Answering System Project (Group 22)

## Student Information
|Student Name|Student Number|
|------------|--------------|
|Chan Tsz Kin|22013718G|
|Wong Ho Wai|2014591G|
|Kok Tsz Ho Zelca|22004159G|

## Project Repository
- Please visit https://github.com/zelcakok-22004159G/comp5423-extract-qa-chatbot for viewing the readme.

## Project Structure
|Folder Name|Description|
|-----------|-----------|
|model-training|Contains the Jupyter notebooks for running on Google Colab|
|backend|Backend server code of the system|
|web|Frontend server code of the system|

## Prerequisite
- NodeJS version >= 12.0.0
    - Installed the `Yarn` library
- Python version >= 3.8
    - Installed the `venv` library
- Make sure the following files are located in the `data` folder
    - train-v2.0.json
    - dev-v2.0.json

## Suggested Hyperparameters
|Variable Name|Value|
|-------------|-----|
|Epoch|2|
|Batch size|24|
|Learning rate|3e-5|

## Training Model
|Filename|Description|
|-----------|-----------|
|data|The simplified version of SQuAD2 training and evaluating dataset for local debugging|
|training-local.py|The training file for debugging (CPU, Batch size = 1, Epoch = 1)|
|evaluate-local.py|The evaluating file for debugging|
|training-on-google-colab.ipynb|The Jupyter notebook to train the QA model|
|evaluate-on-google-colab.ipynb|The Jupyter notebook to evaluate the QA 
model|
|requirements.txt|Includes the dependencies|

### Start training and evaluating locally
```bash
# Navigate to model-training folder
cd model-training

# Create the virtual env
python -m venv venv

# Activate the virutal env
# model-training and backend share the same set of libraries
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Make sure you have activated the virtual environment
source venv/bin/activate

# Please replace the SQuAD2 files in the data/squad folder to gain the reported performace
python training-local.py

# Two folders will be created afte the completion of the training process
# checkpoints and outputs

# Edit the evaluate-local.py and update the variable `model_output_name` to the desired checkpoint
# e.g. model_output_name = "bert-base-squad2-uncased-epoch-1"

python evaluate-local.py

# The evaluation result will be created in the `evaluate-result` folder after the completion of the evaluation.
```


## Starting System Backend
```bash
# Navigate to backend folder
cd backend

# Create the virtual env
python -m venv venv

# Activate the virutal env
# model-training and backend share the same set of libraries
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Make sure you have activated the virtual environment
source venv/bin/activate

# Start the API server which listens on port 9000
python index.py
```

## Starting System Webpage
```bash
# Navigate to web folder
cd web

# Install the dependencies
yarn install

# Start the web server which listens on port 8080
yarn serve
```

