"""YOLO26 training on a laptop (NVIDIA GPU if available, otherwise CPU), for any dataset exported from Roboflow.

Put this file in the folder where you unzipped the Roboflow zip,
next to data.yaml and the train/ valid/ test/ folders, then run:
    python train26.py
The class names are read automatically from data.yaml.
The result is saved in runs/yolo26 (it does not touch YOLOv8's runs/model).
"""
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
EPOCHS = 50  # same as YOLOv8, so the comparison is fair
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # 0 = NVIDIA GPU


def read_classes():
    original = HERE / "data.yaml"
    if not original.is_file():
        raise SystemExit(
            "Cannot find data.yaml. Put train26.py next to data.yaml "
            "and the train/ valid/ test/ folders from the Roboflow zip."
        )
    data = yaml.safe_load(original.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        raise SystemExit("data.yaml does not contain the class names (names).")
    return names


def write_yaml(names):
    folders = {"train": "train/images", "val": "valid/images"}
    if (HERE / "test" / "images").is_dir():
        folders["test"] = "test/images"
    for folder in folders.values():
        if not (HERE / folder).is_dir():
            raise SystemExit(f"Cannot find the folder {folder} next to train26.py.")

    local_yaml = HERE / "data_local.yaml"
    content = {"path": HERE.as_posix(), **folders, "nc": len(names), "names": names}
    local_yaml.write_text(yaml.safe_dump(content, allow_unicode=True), encoding="utf-8")
    return str(local_yaml), "test" in folders


def main():
    names = read_classes()
    print(f"Classes found: {names}")
    print("Training on:", torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU (no GPU)")
    data, has_test = write_yaml(names)

    model = YOLO("yolo26n.pt")
    result = model.train(
        data=data,
        epochs=EPOCHS,
        imgsz=640,
        device=DEVICE,
        workers=0,
        batch=16,
        project=str(HERE / "runs"),
        name="yolo26",  # separate folder, so it does not overwrite YOLOv8's runs/model
        exist_ok=True,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"\nTrained model: {best}")

    if has_test:
        final_model = YOLO(str(best))
        final_model.val(data=data, split="test", device=DEVICE, workers=0)
        pred = final_model.predict(
            str(HERE / "test" / "images"), save=True, conf=0.4, device=DEVICE
        )
        print(f"Test images with boxes: {pred[0].save_dir}")


if __name__ == "__main__":
    main()
