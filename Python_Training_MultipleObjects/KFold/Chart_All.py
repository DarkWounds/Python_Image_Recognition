import os
import glob
import matplotlib.pyplot as plt
import numpy as np

FOLDER = os.path.dirname(os.path.abspath(__file__))
MODELS = ["YOLOv8", "YOLO26", "YOLO12"]


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
        elif key.endswith("_yolo12"):
            metric, model = key[:-len("_yolo12")], "YOLO12"
        elif key.endswith("_yolov8"):
            metric, model = key[:-len("_yolov8")], "YOLOv8"
        elif key.endswith("26"):
            metric, model = key[:-2], "YOLO26"
        elif key.endswith("12"):
            metric, model = key[:-2], "YOLO12"
        elif key.endswith("8"):
            metric, model = key[:-1], "YOLOv8"
        else:
            continue
        table.setdefault(metric, {})[model] = value
    return table


def collect_folds():
    files = sorted(glob.glob(os.path.join(FOLDER, "Values_fold*.txt")))
    if not files:
        raise SystemExit("Nu gasesc Values_fold*.txt. Ruleaza intai Values_All.bat.")
    print("Fisiere citite:", [os.path.basename(f) for f in files])

    # all_vals[metrica][model] = [valoare fold1, valoare fold2, ...]
    all_vals = {}
    for path in files:
        for metric, per_model in split_by_model(read_values(path)).items():
            for model, v in per_model.items():
                all_vals.setdefault(metric, {}).setdefault(model, []).append(v)
    return all_vals, len(files)


def mean_std(vals):
    # ddof=0 -> std de populatie (impartim la N), formula sigma = sqrt(sum((x - mu)^2) / N)
    return float(np.mean(vals)), float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0


def bar_chart(ax, stats, metrics, title, unit):
    metrics = [m for m in metrics if m in stats]
    x = np.arange(len(metrics))
    n = len(MODELS)
    width = 0.8 / n

    for i, model in enumerate(MODELS):
        means = [stats[m].get(model, (0, 0))[0] for m in metrics]
        stds = [stats[m].get(model, (0, 0))[1] for m in metrics]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model)
        ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)


def main():
    all_vals, n_folds = collect_folds()
    stats = {metric: {model: mean_std(v) for model, v in per_model.items()}
             for metric, per_model in all_vals.items()}

    # salvare valori medie +- std
    with open(os.path.join(FOLDER, "Values_mean_std.txt"), "w", encoding="utf-8") as f:
        print(f"Folds: {n_folds}  (media +- std, std de populatie, impartit la N)", file=f)
        for metric, per_model in stats.items():
            for model in MODELS:
                if model in per_model:
                    m, s = per_model[model]
                    print(f"{metric} {model}: {m:.4f} +- {s:.4f}", file=f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"{' vs '.join(MODELS)}  -  media +- std pe {n_folds} fold-uri (K-Fold)", fontsize=14)

    bar_chart(axes[0, 0], stats, ["mAP50-95", "precision", "recall"], "Accuracy", "value (0-1)")
    bar_chart(axes[0, 1], stats, ["preprocess", "inference", "postprocess", "total_latency"], "Speed per image", "ms")
    bar_chart(axes[1, 0], stats, ["FPS"], "FPS (higher = better)", "images / s")

    ax_t = axes[1, 1]
    ax_t.axis("off")
    rows = []
    for metric, per_model in stats.items():
        row = [metric]
        for model in MODELS:
            m, s = per_model.get(model, (0, 0))
            row.append(f"{m:.4f} ± {s:.4f}")
        rows.append(row)
    t = ax_t.table(cellText=rows, colLabels=["Metric"] + MODELS, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)
    ax_t.set_title("Media ± std")

    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "Comparison_KFold.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
