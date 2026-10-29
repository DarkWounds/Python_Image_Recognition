import os
import matplotlib.pyplot as plt
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(FOLDER, "Values.txt")


def read_values(path):
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            try:
                values[key] = float(value)
            except ValueError:
                print("Skipping line:", line)
    return values


def split_by_model(values):
    table = {}
    for key, value in values.items():
        if key.endswith("_yolo26"):
            metric, model = key[:-len("_yolo26")], "YOLO26"
        elif key.endswith("_yolov8"):
            metric, model = key[:-len("_yolov8")], "YOLOv8"
        elif key.endswith("26"):
            metric, model = key[:-2], "YOLO26"
        elif key.endswith("8"):
            metric, model = key[:-1], "YOLOv8"
        else:
            continue
        table.setdefault(metric, {})[model] = value
    return table


def bar_chart(ax, table, metrics, title, unit):
    metrics = [m for m in metrics if m in table]
    x = np.arange(len(metrics))
    width = 0.35
    v8 = [table[m].get("YOLOv8", 0) for m in metrics]
    v26 = [table[m].get("YOLO26", 0) for m in metrics]

    b1 = ax.bar(x - width / 2, v8, width, label="YOLOv8")
    b2 = ax.bar(x + width / 2, v26, width, label="YOLO26")
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)


def main():
    table = split_by_model(read_values(TXT_PATH))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("YOLOv8 vs YOLO26", fontsize=14)

    bar_chart(axes[0, 0], table, ["mAP50-95", "precision", "recall"], "Accuracy", "value (0-1)")
    bar_chart(axes[0, 1], table, ["preprocess", "inference", "postprocess", "total_latency"], "Speed per image", "ms")
    bar_chart(axes[1, 0], table, ["FPS"], "FPS (higher = better)", "images / s")

    # table with all the values
    ax_t = axes[1, 1]
    ax_t.axis("off")
    rows = [[m, f"{v.get('YOLOv8', 0):.4f}", f"{v.get('YOLO26', 0):.4f}"] for m, v in table.items()]
    t = ax_t.table(cellText=rows, colLabels=["Metric", "YOLOv8", "YOLO26"], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.3)
    ax_t.set_title("All values")

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "Comparison.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()