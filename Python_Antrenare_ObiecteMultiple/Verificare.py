from ultralytics import YOLO

FOLDER = r"D:\Python_Image_Recognition\Python_Antrenare_ObiecteMultiple"
CALE_TXT = FOLDER + r"\Valori.txt"
DATA = FOLDER + r"\data_local.yaml"


def procent_mai_mare(a, b):
    return (a - b) / b * 100


def main():
    model8 = YOLO(FOLDER + r"\runs\yolov8n\weights\best.pt")
    model26 = YOLO(FOLDER + r"\runs\yolo26\weights\best.pt")

    metrics8 = model8.val(data=DATA, split="test", batch=1, workers=0)
    metrics26 = model26.val(data=DATA, split="test", batch=1, workers=0)

    map8 = float(metrics8.box.map)
    map26 = float(metrics26.box.map)

    print("mAP50-95 v8:", map8)
    print("mAP50-95 26:", map26)

    with open(CALE_TXT, "w", encoding="utf-8") as f:
        print(f"mAP50-95 yolov8: {map8:.4f}", file=f)
        print(f"mAP50-95 yolo26: {map26:.4f}", file=f)
        if map8 > map26:
            print(f"E mai mare map50-95 yolov8 cu: {procent_mai_mare(map8, map26):.2f}%", file=f)
        elif map8 < map26:
            print(f"E mai mare map50-95 yolo26 cu: {procent_mai_mare(map26, map8):.2f}%", file=f)
        else:
            print("Sunt egale", file=f)

    print("Scris in:", CALE_TXT)


if __name__ == "__main__":
    main()