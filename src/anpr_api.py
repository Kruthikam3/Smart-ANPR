from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import uvicorn

# 🔥 import your processor (adjust path if needed)
from main import processor

app = FastAPI()


@app.get("/")
def home():
    return {"message": "ANPR API running"}


@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    try:
        # ✅ READ IMAGE
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Invalid image"}

        # 🔥 FORCE OCR EVERY TIME
        processor.process_every_n_frames = 1

        # 🔥 RUN DETECTION (same as main.py)
        detections = processor.process_frame(frame, frame_id=0)

        print("\n========================")
        print("RAW DETECTIONS:", detections)
        print("========================\n")

        # ✅ FORMAT RESPONSE
        plates = []
        for det in detections:
            if hasattr(det, "detection_type") and "plate" in det.detection_type:
                plates.append({
                    "text": str(det.text),
                    "confidence": float(det.confidence)
                })

        return {
            "plates": plates,
            "raw_count": len(detections)
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("anpr_api:app", host="0.0.0.0", port=8001, reload=True)

# from fastapi import FastAPI, UploadFile, File
# import cv2
# import numpy as np
# import tempfile

# from main import ANPRProcessor

# app = FastAPI()

# processor = ANPRProcessor()
# processor.load_models()

# @app.post("/detect-plate")
# async def detect_plate(file: UploadFile = File(...)):

#     # read image
#     contents = await file.read()
#     np_arr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     # ✅ check
#     if frame is None:
#         return {"error": "Invalid image"}

#     # ✅ RGBA fix
#     if len(frame.shape) == 3 and frame.shape[2] == 4:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

#     # ✅ VERY IMPORTANT FIX 1: color conversion
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # ✅ VERY IMPORTANT FIX 2: resize (match model input)
#     frame = cv2.resize(frame, (640, 640))

#     # ✅ DEBUG (see what API is actually sending)
#     cv2.imwrite("debug_api_input.jpg", frame)

#     # call model
#     detections = processor.process_frame(frame, frame_id=0)

#     print("DETECTIONS:", detections)

#     plates = []

#     # ✅ STEP 4: process detections
#     for det in detections:
#         if "plate" in det.detection_type:

#             print("TEXT:", det.text)
            
#             plates.append({
#                 "text": str(det.text),
#                 "confidence": float(det.confidence)
#             })

#     return {"plates": plates}