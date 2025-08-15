from fastapi import FastAPI, UploadFile, Form
from main import run as run_pipeline
import tempfile, shutil

app = FastAPI()

@app.post("/infer")
async def infer(image: UploadFile, query: str = Form(...)):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        shutil.copyfileobj(image.file, f); img_path = f.name
    # chạy pipeline
    run_pipeline(img_path, query)
    # trả file kết quả
    return {"result": "output.jpg"}
