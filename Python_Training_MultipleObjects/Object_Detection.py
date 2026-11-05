from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FOLDER = Path(__file__).resolve().parent
FolderImg = FOLDER / "Objects_To_Detect"
FolderSave = FOLDER / "Detected_Objects"


model8 = YOLO(str(FOLDER / "runs" / "yolov8n" / "weights" / "best.pt"))
model26 = YOLO(str(FOLDER / "runs" / "yolo26" / "weights" / "best.pt"))
model12 = YOLO(str(FOLDER / "runs" / "yolo12" / "weights" / "best.pt"))

imagesR = [str(FolderImg / "R1.jpeg"), str(FolderImg / "R2.jpeg"), str(FolderImg / "R3.jpeg")]  # Rubiks cube
imagesH = [str(FolderImg / "H1.jpeg"), str(FolderImg / "H2.jpeg"), str(FolderImg / "H3.jpeg")]  # Hand cream
imagesS = [str(FolderImg / "S1.jpeg"), str(FolderImg / "S2.jpeg"), str(FolderImg / "S3.jpeg")]  # Super glue

# Run batched inference on a list of images
resultsRubiks8 = model8(imagesR)  # return a list of Results objects
resultsRubiks26 = model26(imagesR)
resultsRubiks12 = model12(imagesR)

resultsHand8 = model8(imagesH)
resultsHand26 = model26(imagesH)
resultsHand12 = model12(imagesH)

resultsSuper8 = model8(imagesS)
resultsSuper26 = model26(imagesS)
resultsSuper12 = model12(imagesS)

# Process results list

# Rubiks cube
for i, result in enumerate(resultsRubiks8):
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.save(filename=str(FolderSave / f"resultRubik8_{i}.jpg"))  # save to disk

for i, result in enumerate(resultsRubiks26):
    boxes = result.boxes
    result.save(filename=str(FolderSave / f"resultRubik26_{i}.jpg"))

for i, result in enumerate(resultsRubiks12):
    boxes = result.boxes
    result.save(filename=str(FolderSave / f"resultRubik12_{i}.jpg"))

# Hand cream
for i, result in enumerate(resultsHand8):
    boxes = result.boxes
    result.save(filename=str(FolderSave / f"resultHand8_{i}.jpg"))

for i, result in enumerate(resultsHand26):
    boxes = result.boxes
    result.save(filename=str(FolderSave / f"resultHand26_{i}.jpg"))

for i, result in enumerate(resultsHand12):
    boxes = result.boxes
    result.show()
    result.save(filename=str(FolderSave / f"resultHand12_{i}.jpg"))

# Super glue
for i, result in enumerate(resultsSuper8):
    boxes = result.boxes
    result.save(filename=str(FolderSave / f"resultSuper8_{i}.jpg"))

for i, result in enumerate(resultsSuper26):
    boxes = result.boxes
    result.save(filename=str(FolderSave / f"resultSuper26_{i}.jpg"))

for i, result in enumerate(resultsSuper12):
    boxes = result.boxes
    result.save(filename=str(FolderSave / f"resultSuper12_{i}.jpg"))