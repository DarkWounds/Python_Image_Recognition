import os
import matplotlib.pyplot as plt
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
CALE_TXT = os.path.join(FOLDER, "Valori.txt")


def citeste_valori(cale):
    valori = {}
    with open(cale, encoding="utf-8") as f:
        for linie in f:
            linie = linie.strip()
            if ": " not in linie:
                continue
            cheie, valoare = linie.split(": ", 1)
            try:
                valori[cheie] = float(valoare)
            except ValueError:
                print("Sar peste linia:", linie)
    return valori


def separa_pe_modele(valori):
    tabel = {}
    for cheie, valoare in valori.items():
        if cheie.endswith("_yolo26"):
            metrica, model = cheie[:-len("_yolo26")], "YOLO26"
        elif cheie.endswith("_yolov8"):
            metrica, model = cheie[:-len("_yolov8")], "YOLOv8"
        elif cheie.endswith("26"):
            metrica, model = cheie[:-2], "YOLO26"
        elif cheie.endswith("8"):
            metrica, model = cheie[:-1], "YOLOv8"
        else:
            continue
        tabel.setdefault(metrica, {})[model] = valoare
    return tabel


def grafic_bare(ax, tabel, metrici, titlu, unitate):
    metrici = [m for m in metrici if m in tabel]
    x = np.arange(len(metrici))
    latime = 0.35
    v8 = [tabel[m].get("YOLOv8", 0) for m in metrici]
    v26 = [tabel[m].get("YOLO26", 0) for m in metrici]

    b1 = ax.bar(x - latime / 2, v8, latime, label="YOLOv8")
    b2 = ax.bar(x + latime / 2, v26, latime, label="YOLO26")
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrici)
    ax.set_title(titlu)
    ax.set_ylabel(unitate)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)


def main():
    tabel = separa_pe_modele(citeste_valori(CALE_TXT))

    fig, axe = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("YOLOv8 vs YOLO26", fontsize=14)

    grafic_bare(axe[0, 0], tabel, ["mAP50-95", "precision", "recall"], "Precizie", "valoare (0-1)")
    grafic_bare(axe[0, 1], tabel, ["preprocess", "inference", "postprocess", "latenta_totala"], "Viteza per imagine", "ms")
    grafic_bare(axe[1, 0], tabel, ["FPS"], "FPS (mai mare = mai bine)", "imagini / s")

    # tabel cu toate valorile
    ax_t = axe[1, 1]
    ax_t.axis("off")
    randuri = [[m, f"{v.get('YOLOv8', 0):.4f}", f"{v.get('YOLO26', 0):.4f}"] for m, v in tabel.items()]
    t = ax_t.table(cellText=randuri, colLabels=["Metrica", "YOLOv8", "YOLO26"], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.3)
    ax_t.set_title("Toate valorile")

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "Comparatie.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()