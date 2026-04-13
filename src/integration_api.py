from fastapi import FastAPI, UploadFile, File
import requests

app = FastAPI()

FRS_URL = "http://localhost:5000/recognize"   # update if needed
ANPR_URL = "http://localhost:8001/detect-plate"

@app.post("/detect-all")
async def detect_all(file: UploadFile = File(...)):

    image_bytes = await file.read()

    # Call ANPR
    anpr_response = requests.post(
        ANPR_URL,
        files={"file": image_bytes}
    )
    plates = anpr_response.json().get("plates")

    # Call FRS
    frs_response = requests.post(
        FRS_URL,
        files={"file": image_bytes}
    )
    name = frs_response.json().get("name")

    return {
        "person": name,
        "vehicle": plates
    }