import torch
import gradio as gr
import pandas as pd
from model import LSTMModel

# ------------------------------
# Modeli yükle
model = LSTMModel()
model.load_state_dict(torch.load("usdtry_lstm.pth", map_location="cpu"))
model.eval()

# ------------------------------
# Normalizasyon için veri bilgisi (train.py ile aynı)
df = pd.read_csv("USD_to_TL_currency_2010_2024_years.csv")
prices = df["currency_usd_to_tl"].values.astype(float)

min_price = prices.min()
max_price = prices.max()

# ------------------------------
# Tahmin fonksiyonu
def predict_usd_try(inputs):
    try:
        values = [float(x.strip()) for x in inputs.split(",")]

        if len(values) != 4:
            return "Lütfen tam olarak 4 günün USD/TRY değerini giriniz."

        # Normalize
        values_norm = [(v - min_price) / (max_price - min_price) for v in values]

        x = torch.tensor([values_norm], dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            pred_norm = model(x)

        # De-normalize
        pred_real = pred_norm.item() * (max_price - min_price) + min_price

        return f"Tahmin edilen bir sonraki gün USD/TRY: {pred_real:.4f}"

    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# ------------------------------
# Gradio arayüzü
interface = gr.Interface(
    fn=predict_usd_try,
    inputs=gr.Textbox(
        placeholder="Örnek: 29.10, 29.25, 29.40, 29.55",
        label="Son 4 günün USD/TRY değerleri"
    ),
    outputs=gr.Textbox(label="Model Tahmini"),
    title="USD/TRY Döviz Kuru Tahmini (LSTM)",
    description="Son 4 günün USD/TRY değerlerini girerek bir sonraki günün tahmini değerini elde edebilirsiniz."
)

# ------------------------------
# Uygulamayı başlat
if __name__ == "__main__":
    interface.launch(share=True)
