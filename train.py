import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import LSTMModel

# ------------------------------
# Veri yükleme
df = pd.read_csv("USD_to_TL_currency_2010_2024_years.csv")

# Sadece fiyat kolonu
prices = df["currency_usd_to_tl"].values.astype(float)

# ------------------------------
# Normalize
min_price = prices.min()
max_price = prices.max()
prices_norm = (prices - min_price) / (max_price - min_price)

# ------------------------------
# Sequence oluşturma
seq_length = 4
X, y = [], []

for i in range(len(prices_norm) - seq_length):
    X.append(prices_norm[i:i+seq_length])
    y.append(prices_norm[i+seq_length])

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# ------------------------------
# Model
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ------------------------------
# Eğitim
epochs = 1000
model.train()

loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# ------------------------------
# Loss grafiği
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Model Eğitim Sürecinde Loss Değişimi")
plt.grid(True)
plt.savefig("loss_graph.png")   # 👈 RAPOR İÇİN
plt.show()

# ------------------------------
# Test (son 4 güne bakarak)
model.eval()
with torch.no_grad():
    test_seq = X[-1].unsqueeze(0)
    pred_norm = model(test_seq)

pred_real = pred_norm.item() * (max_price - min_price) + min_price
print("Tahmin edilen bir sonraki gün USD/TRY:", pred_real)

# ------------------------------
# Gerçek vs Tahmin (son 100 gün)
predictions = []

with torch.no_grad():
    for i in range(len(X)-100, len(X)):
        pred = model(X[i].unsqueeze(0))
        predictions.append(pred.item())

# De-normalize
predictions = np.array(predictions) * (max_price - min_price) + min_price
real_values = prices[-100:]

plt.figure(figsize=(8, 5))
plt.plot(real_values, label="Gerçek Değerler")
plt.plot(predictions, label="Tahmin Edilen Değerler")
plt.xlabel("Gün")
plt.ylabel("USD/TRY")
plt.title("Gerçek ve Tahmin Edilen USD/TRY Değerleri")
plt.legend()
plt.grid(True)
plt.savefig("real_vs_pred.png")   # 👈 RAPOR İÇİN
plt.show()

# ------------------------------
# Modeli kaydet
torch.save(model.state_dict(), "usdtry_lstm.pth")
print("Model kaydedildi.")
