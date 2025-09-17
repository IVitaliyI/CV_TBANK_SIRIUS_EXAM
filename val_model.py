import pandas as pd
import matplotlib.pyplot as plt

# читаем лог обучения
df = pd.read_csv("runs/detect/train3/results.csv")

# строим графики
plt.figure(figsize=(10,5))
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("YOLO Training Metrics")
plt.legend()
plt.show()
