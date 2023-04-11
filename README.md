# COMP5431 NLP Project
## Extract Question Answering Chatbot

### How to start project
```bash
# Please install NodeJS + npm first
#
# npm insatll -g yarn        // if yarn is not installed

// Start Web
cd web && yarn && yarn serve

// Web will start on port 8080

// Start Backend
cd backend 
python3 -m venv venv            // create virtual env. Command python3 or python is okay
source venv/bin/activate        // activate virtual env
pip install -r requirements.txt // install libraries
python3 index.py                // run code. Command python3 or python is okay

// API will start on port 9000
```