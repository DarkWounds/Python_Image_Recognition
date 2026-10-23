"""Antrenare YOLO26 pe laptop (GPU NVIDIA dacă există, altfel CPU), pentru orice dataset exportat din Roboflow.

Pune acest fișier în folderul unde ai dezarhivat zip-ul de la Roboflow,
lângă data.yaml și folderele train/ valid/ test/, apoi rulează:
    python antrenare26.py
Numele claselor sunt citite automat din data.yaml.
Rezultatul se salvează în runs/yolo26 (nu atinge runs/model de la YOLOv8).
"""
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO

AICI = Path(__file__).resolve().parent
EPOCI = 50  # la fel ca la YOLOv8, ca testul să fie corect
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # 0 = placa video NVIDIA


def citeste_clase():
    original = AICI / "data.yaml"
    if not original.is_file():
        raise SystemExit(
            "Nu găsesc data.yaml. Pune antrenare26.py lângă data.yaml "
            "și folderele train/ valid/ test/ din zip-ul de la Roboflow."
        )
    date = yaml.safe_load(original.read_text(encoding="utf-8"))
    nume = date.get("names")
    if isinstance(nume, dict):
        nume = [nume[k] for k in sorted(nume)]
    if not nume:
        raise SystemExit("data.yaml nu conține numele claselor (names).")
    return nume


def scrie_yaml(nume):
    foldere = {"train": "train/images", "val": "valid/images"}
    if (AICI / "test" / "images").is_dir():
        foldere["test"] = "test/images"
    for folder in foldere.values():
        if not (AICI / folder).is_dir():
            raise SystemExit(f"Nu găsesc folderul {folder} lângă antrenare26.py.")

    yaml_local = AICI / "data_local.yaml"
    continut = {"path": AICI.as_posix(), **foldere, "nc": len(nume), "names": nume}
    yaml_local.write_text(yaml.safe_dump(continut, allow_unicode=True), encoding="utf-8")
    return str(yaml_local), "test" in foldere


def main():
    nume = citeste_clase()
    print(f"Clase găsite: {nume}")
    print("Antrenez pe:", torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU (fără GPU)")
    data, are_test = scrie_yaml(nume)

    model = YOLO("yolo26n.pt")
    rezultat = model.train(
        data=data,
        epochs=EPOCI,
        imgsz=640,
        device=DEVICE,
        workers=0,
        batch=16,
        project=str(AICI / "runs"),
        name="yolo26",  # folder separat, ca să nu suprascrie runs/model de la YOLOv8
        exist_ok=True,
    )

    best = Path(rezultat.save_dir) / "weights" / "best.pt"
    print(f"\nModel antrenat: {best}")

    if are_test:
        model_final = YOLO(str(best))
        model_final.val(data=data, split="test", device=DEVICE, workers=0)
        pred = model_final.predict(
            str(AICI / "test" / "images"), save=True, conf=0.4, device=DEVICE
        )
        print(f"Pozele de test cu dreptunghiuri: {pred[0].save_dir}")


if __name__ == "__main__":
    main()
