from fastapi import FastAPI, Query
from bson import json_util
import json

from videoservice import get_videos_by_creator

app = FastAPI()

@app.get("/api/data")
def get_data(creatorId: str = Query(...)):
    raw_data = get_videos_by_creator(creator_id=creatorId)

    if raw_data is None:
        return {
            "success": False,
            "message": "Invalid creatorId format"
        }

    data = json.loads(json_util.dumps(raw_data))

    return {
        "success": True,
        "data": data
    }