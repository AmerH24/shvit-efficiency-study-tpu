import csv
import json
import os


EXPERIMENTS = {
    "ratio_1_8": (
        "results_tpu/"
        "ratio_1_8/log.txt"
    ),
    "ratio_default": (
        "results_tpu/"
        "ratio_default/log.txt"
    ),
    "ratio_1_2": (
        "results_tpu/"
        "ratio_1_2/log.txt"
    ),
    "progressive": (
        "results_tpu/"
        "progressive/log.txt"
    ),
}


def read_best_result(path):
    best = None

    with open(path, "r") as file:
        for line in file:
            row = json.loads(line)

            if (
                best is None
                or row["test_acc1"]
                > best["test_acc1"]
            ):
                best = row

    return best


def main():
    rows = []

    for name, path in (
        EXPERIMENTS.items()
    ):
        if not os.path.exists(path):
            print(
                f"Missing: {path}"
            )

            continue

        best = read_best_result(
            path
        )

        rows.append({
            "variant": name,
            "best_epoch": (
                best["epoch"]
            ),
            "top1_accuracy": (
                best["test_acc1"]
            ),
            "epoch_time_seconds": (
                best[
                    "epoch_time_seconds"
                ]
            ),
        })

    if not rows:
        print(
            "No TPU results found."
        )

        return

    os.makedirs(
        "results_tpu",
        exist_ok=True,
    )

    output_path = (
        "results_tpu/"
        "accuracy_tpu.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "\nTPU results:"
    )

    for row in rows:
        print(
            f"{row['variant']:<18} "
            f"Top-1="
            f"{row['top1_accuracy']:.2f}% "
            f"epoch="
            f"{row['best_epoch']}"
        )


if __name__ == "__main__":
    main()