
import json

from datetime import datetime
from langchain.schema.messages import HumanMessage, AIMessage

def save_chat_history_json(chat_history, file_path):
    json_data = [message.dict() for message in chat_history] 
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)

def load_chat_history_json(file_path):
    with open(file_path, "r") as f:
        json_data = json.load(f)
        messages = [HumanMessage(**message) if message["type"] == "human" else AIMessage(**message) for message in json_data]
        return messages



def get_time_stamp():
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")
