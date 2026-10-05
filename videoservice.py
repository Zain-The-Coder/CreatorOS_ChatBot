from fastapi import FastAPI, Query
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import json_util, ObjectId
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

app = FastAPI()

MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["creatorOS"]
collection = db["videos"]


def get_videos_by_creator(creator_id: str):
    """
    Creator ki saari videos MongoDB se fetch karta hai.
    Return: list of video documents, ya None agar invalid ID
    """
    try:
        creator_object_id = ObjectId(creator_id)
    except Exception:
        return None

    raw_data = list(collection.find({"creatorId": creator_object_id}))
    return raw_data
