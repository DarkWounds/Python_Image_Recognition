"""Antrenare YOLO pentru clasa CASTI, pe laptop (CPU).

Pune acest fișier în același folder cu data.yaml și folderele train/ valid/ test/
(adică în folderul unde ai dezarhivat casti_dataset.zip), apoi rulează:
    python antrenare.py
"""
from pathlib import Path

from ultralytics import YOLO

AICI = Path(__file__).resolve().parent
EPOCI = 50  # pune 30 dacă vrei să termine mai repede


def scrie_yaml():
    yaml_local = AICI / "data_local.yaml"
    yaml_local.write_text(
        f"path: {AICI.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['CASTI']\n",
        encoding="utf-8",
    )
    return str(yaml_local)


def main():
    for folder in ("train/images", "valid/images", "test/images"):
        if not (AICI / folder).is_dir():
            raise SystemExit(
                f"Nu găsesc folderul {folder}. Pune antrenare.py lângă data.yaml "
                "și folderele train/ valid/ test/ din casti_dataset.zip."
            )

    data = scrie_yaml()

    model = YOLO("yolov8n.pt")
    rezultat = model.train(
        data=data,
        epochs=EPOCI,
        imgsz=640,
        device="cpu",
        workers=0,
        project=str(AICI / "runs"),
        name="casti",
        exist_ok=True,
    )

    best = Path(rezultat.save_dir) / "weights" / "best.pt"
    print(f"\nModel antrenat: {best}")

    model_final = YOLO(str(best))
    model_final.val(data=data, split="test", device="cpu", workers=0)
    pred = model_final.predict(
        str(AICI / "test" / "images"), save=True, conf=0.4, device="cpu"
    )
    print(f"Pozele de test cu dreptunghiuri: {pred[0].save_dir}")


if __name__ == "__main__":
    main()
