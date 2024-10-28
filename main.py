from model import model_pipeline
from typing import Union
from fastapi import FastAPI, UploadFile
from PIL import Image

app = FastAPI()

@app.get('/')
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post('/ask')
def ask(text: str, image: UploadFile):

    image = Image.open(image.file)
    results = model_pipeline(text, image)

    return {'answer': results}
