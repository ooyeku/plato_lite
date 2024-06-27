# utils/file_utils.py

import base64
import json
from datetime import datetime

def encode_image(file_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def save_json(data: dict, file_path: str):
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

def load_json(file_path: str) -> dict:
    """Load data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

class DateTimeEncoder(json.JSONEncoder):
    """JSON Encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)