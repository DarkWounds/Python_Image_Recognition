from ultralytics import YOLO

FOLDER = r"D:\Python_Image_Recognition\Python_Antrenare_ObiecteMultiple"
CALE_TXT = FOLDER + r"\Tabel\Valori.txt"
DATA = FOLDER + r"\data_local.yaml"


def procent_mai_mare(a, b):
    return (a - b) / b * 100


def main():
    model8 = YOLO(FOLDER + r"\runs\yolov8n\weights\best.pt")
    model26 = YOLO(FOLDER + r"\runs\yolo26\weights\best.pt")

    metrics8 = model8.val(data=DATA, split="test", batch=1, workers=0)
    metrics26 = model26.val(data=DATA, split="test", batch=1, workers=0)

    latenta8 = float(float(metrics8.speed['preprocess']) + float(metrics8.speed['inference']) + float(metrics8.speed['postprocess']))
    latenta26 = float(metrics26.speed['preprocess']) + float(metrics26.speed['inference']) + float(metrics26.speed['postprocess'])

    with open(CALE_TXT, "w", encoding="utf-8") as f:
        print(f"mAP50-95_yolov8: {float(metrics8.box.map):.4f}", file=f)
        print(f"mAP50-95_yolo26: {float(metrics26.box.map):.4f}", file=f)
        print(f"precision8: {float(metrics8.box.mp):.4f}", file=f)
        print(f"precision26: {float(metrics26.box.mp):.4f}", file=f)
        print(f"recall8: {float(metrics8.box.mr):.4f}", file=f)
        print(f"recall26: {float(metrics26.box.mr):.4f}", file=f)
        print(f"preprocess8: {float(metrics8.speed['preprocess']):.4f}", file=f)
        print(f"preprocess26: {float(metrics26.speed['preprocess']):.4f}", file=f)
        print(f"inference8: {float(metrics8.speed['inference']):.4f}", file=f)
        print(f"inference26: {float(metrics26.speed['inference']):.4f}", file=f)
        print(f"postprocess8: {float(metrics8.speed['postprocess']):.4f}", file=f)
        print(f"postprocess26: {float(metrics26.speed['postprocess']):.4f}", file=f)
        print(f"latenta_totala8: {latenta8:.4f}", file=f)
        print(f"latenta_totala26: {latenta26:.4f}", file=f)
        print(f"FPS8: {float(1000/latenta8):.4f}", file=f)
        print(f"FPS26: {float(1000/latenta26):.4f}", file=f)

    print("Scris in:", CALE_TXT)

if __name__ == "__main__":
    main()