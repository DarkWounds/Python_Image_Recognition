from pathlib import Path

from ultralytics import YOLO

FOLDER = Path(__file__).resolve().parent
TXT_PATH = FOLDER / "Chart" / "Values.txt"
DATA = str(FOLDER / "data_local.yaml")


def percent_higher(a, b):
    return (a - b) / b * 100


def main():
    model8 = YOLO(str(FOLDER / "runs" / "yolov8n" / "weights" / "best.pt"))
    model26 = YOLO(str(FOLDER / "runs" / "yolo26" / "weights" / "best.pt"))

    metrics8 = model8.val(data=DATA, split="test", batch=1, workers=0)
    metrics26 = model26.val(data=DATA, split="test", batch=1, workers=0)

    latency8 = float(metrics8.speed['preprocess']) + float(metrics8.speed['inference']) + float(metrics8.speed['postprocess'])
    latency26 = float(metrics26.speed['preprocess']) + float(metrics26.speed['inference']) + float(metrics26.speed['postprocess'])

    with open(TXT_PATH, "w", encoding="utf-8") as f:
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
        print(f"total_latency8: {latency8:.4f}", file=f)
        print(f"total_latency26: {latency26:.4f}", file=f)
        print(f"FPS8: {float(1000 / latency8):.4f}", file=f)
        print(f"FPS26: {float(1000 / latency26):.4f}", file=f)

    print("Written to:", TXT_PATH)


if __name__ == "__main__":
    main()
