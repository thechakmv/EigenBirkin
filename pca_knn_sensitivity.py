from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from run_pipeline import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PCA-dimension and KNN-neighbor sensitivity sweep for Birkin vs not-Birkin classification."
    )
    parser.add_argument(
        "--birkin-dirs",
        nargs="+",
        type=Path,
        default=[Path("Data/Birkin"), Path("Data/birkins")],
        help="Directories containing Birkin images.",
    )
    parser.add_argument(
        "--other-dir",
        type=Path,
        default=Path("Data/other"),
        help="Directory containing non-Birkin images.",
    )
    parser.add_argument("--size", type=int, default=64, help="Image resize size (size x size).")
    parser.add_argument(
        "--components-list",
        nargs="+",
        type=int,
        default=[12, 24, 48],
        help="List of PCA dimensions to evaluate.",
    )
    parser.add_argument(
        "--k-list",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="List of KNN neighbor counts to evaluate.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Evaluation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/pca_knn_sensitivity_report.txt"),
        help="Where to save the sensitivity report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x, y = build_dataset(args.birkin_dirs, args.other_dir, args.size)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    component_values = sorted({c for c in args.components_list if c > 0})
    k_values = sorted({k for k in args.k_list if k > 0})
    if not component_values:
        raise ValueError("--components-list must contain at least one positive integer")
    if not k_values:
        raise ValueError("--k-list must contain at least one positive integer")

    results: list[dict[str, float | int]] = []

    for requested_components in component_values:
        n_components = min(requested_components, x_train.shape[0], x_train.shape[1])
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True, random_state=args.seed)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)

        for k in k_values:
            model = KNeighborsClassifier(n_neighbors=k, weights="distance")
            model.fit(x_train_pca, y_train)
            y_pred = model.predict(x_test_pca)

            accuracy = float((y_pred == y_test).mean())
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            results.append(
                {
                    "requested_components": requested_components,
                    "n_components": n_components,
                    "k": k,
                    "accuracy": accuracy,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                }
            )

    results.sort(key=lambda item: (-float(item["accuracy"]), int(item["n_components"]), int(item["k"])))

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as file:
        file.write("PCA dimension and K sensitivity sweep\n")
        file.write(f"total_samples={len(x)}\n")
        file.write(f"birkin_samples={int(y.sum())}\n")
        file.write(f"not_birkin_samples={len(y) - int(y.sum())}\n")
        file.write(f"train_samples={len(x_train)}\n")
        file.write(f"test_samples={len(x_test)}\n")
        file.write(f"image_size={args.size}x{args.size}\n")
        file.write(f"components_list={component_values}\n")
        file.write(f"k_list={k_values}\n")
        file.write(f"seed={args.seed}\n\n")
        file.write("Results sorted by accuracy (best first):\n")
        file.write("requested_components\tused_components\tk\taccuracy\tTN\tFP\tFN\tTP\n")
        for result in results:
            file.write(
                f"{result['requested_components']}\t{result['n_components']}\t{result['k']}\t"
                f"{result['accuracy']:.4f}\t{result['tn']}\t{result['fp']}\t{result['fn']}\t{result['tp']}\n"
            )

    best = results[0]
    print("Best configuration:")
    print(
        f"  requested_components={best['requested_components']} used_components={best['n_components']} "
        f"k={best['k']} accuracy={best['accuracy']:.4f}"
    )
    print(f"Saved report: {args.report_path}")


if __name__ == "__main__":
    main()
