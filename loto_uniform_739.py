import numpy as np
import pandas as pd


CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7hh_4594_k28.csv"


def load_draws(path):
    df = pd.read_csv(path)
    cols = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
    if all(c in df.columns for c in cols):
        arr = df[cols].copy()
    else:
        arr = df.iloc[:, :7].copy()
    arr = arr.apply(pd.to_numeric, errors="coerce").dropna()
    draws = arr.values.astype(int)
    return draws


def uniform_without_replacement_prediction():
    # Deterministički izbor 7 brojeva najbližih teorijskoj uniformnoj učestalosti
    draws = load_draws(CSV_PATH)
    counts = np.bincount(draws.ravel(), minlength=40)[1:]
    expected = counts.sum() / 39.0
    order = sorted(range(1, 40), key=lambda n: (abs(counts[n - 1] - expected), n))
    pred = np.array(order[:7], dtype=int)
    pred.sort()
    return pred


def main():
    draws = load_draws(CSV_PATH)
    flat = draws.ravel()

    # Empirijska frekvencija po broju 1..39
    counts = np.bincount(flat, minlength=40)[1:]
    probs = counts / counts.sum()

    print("CSV:", CSV_PATH)
    print("Broj izvlacenja:", len(draws))
    print("Teorijska verovatnoca po broju (uniform):", round(1 / 39, 6))
    print("Empirijska min/max verovatnoca:", round(float(probs.min()), 6), round(float(probs.max()), 6))
    print()

    pred = uniform_without_replacement_prediction()
    print("Predicted next loto 7/39 combination:", pred)


if __name__ == "__main__":
    main()

"""
Broj izvlacenja: 4594
Teorijska verovatnoca po broju (uniform): 0.025641
Empirijska min/max verovatnoca: 0.02354 0.028111

Predicted next loto 7/39 combination: 
[ 3  5 13 16 21 24 31]
"""
