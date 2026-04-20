# logger.py
import csv
import matplotlib.pyplot as plt

def log_results(rows, file="results.csv"):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "match_ratio", "detected"])
        writer.writerows(rows)

def plot_results(file="results.csv"):
    attacks = []
    scores = []

    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attacks.append(row["attack"])
            scores.append(float(row["match_ratio"]))

    plt.figure()
    plt.bar(attacks, scores)
    plt.xlabel("Attack")
    plt.ylabel("Match Ratio")
    plt.title("Watermark Robustness")
    plt.show()