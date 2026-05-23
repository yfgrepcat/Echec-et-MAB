import json
import glob

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------
# Chargement de tous les logs
# ---------------------------------

log_files = glob.glob("logs/games_worker_*.jsonl")

rows = []

for log_file in log_files:

    with open(log_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))


# ---------------------------------
# DataFrame
# ---------------------------------

df = pd.DataFrame(rows)

print("\nLoaded rows:")
print(len(df))

print("\nColumns:")
print(df.columns)

print("\nHead:")
print(df.head())


# ---------------------------------
# Reward moyen
# ---------------------------------

print("\nAverage reward:")
print(df["reward"].mean())


# ---------------------------------
# Distribution des bras
# ---------------------------------

print("\nArm distribution:")
print(df["arm"].value_counts())


# ---------------------------------
# Temps moyen par bras
# ---------------------------------

print("\nAverage elapsed time by arm:")
print(
    df.groupby("arm")["elapsed"].mean()
)


# ---------------------------------
# Temps moyen par niveau Stockfish
# ---------------------------------

print("\nAverage reward by Stockfish level:")

print(
    df.groupby("stockfish_level")["reward"].mean()
)


# ---------------------------------
# Reward rolling
# ---------------------------------

plt.figure(figsize=(12, 6))

rolling_reward = (
    df["reward"]
    .rolling(100)
    .mean()
)

plt.plot(rolling_reward)

plt.title("Rolling Reward")

plt.xlabel("Move")

plt.ylabel("Reward")

plt.grid()

plt.show()


# ---------------------------------
# Distribution des bras
# ---------------------------------

plt.figure(figsize=(8, 5))

df["arm"] \
    .value_counts() \
    .sort_index() \
    .plot(kind="bar")

plt.title("Arm Usage")

plt.xlabel("Arm")

plt.ylabel("Count")

plt.grid()

plt.show()


# ---------------------------------
# Temps moyen par bras
# ---------------------------------

plt.figure(figsize=(8, 5))

df.groupby("arm")["elapsed"] \
    .mean() \
    .plot(kind="bar")

plt.title("Average Thinking Time by Arm")

plt.xlabel("Arm")

plt.ylabel("Seconds")

plt.grid()

plt.show()


# ---------------------------------
# Reward par niveau
# ---------------------------------

plt.figure(figsize=(8, 5))

df.groupby("stockfish_level")["reward"] \
    .mean() \
    .plot(kind="bar")

plt.title("Average Reward vs Stockfish Level")

plt.xlabel("Stockfish Level")

plt.ylabel("Reward")

plt.grid()

plt.show()
