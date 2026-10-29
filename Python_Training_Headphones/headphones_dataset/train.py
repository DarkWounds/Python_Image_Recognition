"""YOLO training for the HEADPHONES class, on a laptop (CPU).

Put this file in the same folder as data.yaml and the train/ valid/ test/ folders
(the folder where you unzipped headphones_dataset.zip), then run:
    python train.py
"""
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
EPOCHS = 50  # use 30 if you want it to finish faster


def write_yaml():
    local_yaml = HERE / "data_local.yaml"
    local_yaml.write_text(
        f"path: {HERE.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['Headphones']\n",
        encoding="utf-8",
    )
    return str(local_yaml)


def main():
    for folder in ("train/images", "valid/images", "test/images"):
        if not (HERE / folder).is_dir():
            raise SystemExit(
                f"Cannot find the folder {folder}. Put train.py next to data.yaml "
                "and the train/ valid/ test/ folders from headphones_dataset.zip."
            )

    data = write_yaml()

    model = YOLO("yolov8n.pt")
    result = model.train(
        data=data,
        epochs=EPOCHS,
        imgsz=640,
        device="cpu",
        workers=0,
        project=str(HERE / "runs"),
        name="headphones",
        exist_ok=True,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"\nTrained model: {best}")

    final_model = YOLO(str(best))
    final_model.val(data=data, split="test", device="cpu", workers=0)
    pred = final_model.predict(
        str(HERE / "test" / "images"), save=True, conf=0.4, device="cpu"
    )
    print(f"Test images with boxes: {pred[0].save_dir}")


if __name__ == "__main__":
    main()
