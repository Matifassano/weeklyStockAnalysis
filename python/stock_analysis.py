# import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import telegram
import io
import asyncio
from datetime import datetime, timedelta
import requests

# Configuración
stocks = ["YPF", 'TSM']
bot_token = "8270763519:AAHxGeVI0AwWdTcpCVkGa4ceixHTvDdWHuE"
chat_id = "804366247"

# Descargar datos
def fetch_data(tickers):
    data = {}
    today = datetime.now().strftime('%Y-%m-%d')
    for ticker in tickers:
        df = yf.download(ticker, start='2025-01-01', end=today, auto_adjust=True)
        data[ticker] = df
    return data

# Gráficos
def analyze_and_plot(data):
    plots = {}
    for ticker, df in data.items():
        # Serie semanal (último cierre de cada semana)
        weekly = df['Close'].resample('W-FRI').last().dropna()
        if len(weekly) < 10:
            continue

        # Tendencia: EMAs 20/50 semanas
        ema20 = weekly.ewm(span=20, adjust=False).mean()
        ema50 = weekly.ewm(span=50, adjust=False).mean()

        # Momentum: MACD (12,26,9) semanal
        ema12 = weekly.ewm(span=12, adjust=False).mean()
        ema26 = weekly.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_hist = (macd - macd_sig).squeeze()


        # Momentum: RSI(14) semanal
        delta = weekly.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))

               # Pendiente logarítmica de 13 semanas (% por semana)
        w = 13 if len(weekly) >= 13 else max(2, len(weekly) - 1)
        y = np.log(weekly.tail(w).values)
        x = np.arange(len(y)).reshape(-1, 1)
        lin = LinearRegression().fit(x, y)
        coef = float(np.ravel(lin.coef_)[0])
        slope_w = (np.exp(coef) - 1) * 100


        # Rendimientos recientes
        def pct_change_n(n):
            if len(weekly) > n:
                return float((weekly.iloc[-1] / weekly.iloc[-n-1] - 1) * 100)
            return np.nan

        ret4w = pct_change_n(4)
        ret52w = pct_change_n(52)

        # Figura 3 paneles
        fig = plt.figure(figsize=(9, 7))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 2], hspace=0.1)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(weekly.index, weekly.values, color='black', linewidth=1.5, label='Close (W)')
        ax1.plot(ema20.index, ema20.values, color='tab:blue', linewidth=1.2, label='EMA20W')
        ax1.plot(ema50.index, ema50.values, color='tab:orange', linewidth=1.2, label='EMA50W')
        ax1.set_title(f"{ticker} (Semanal) – Tendencia y Momento", fontsize=12)
        subtitle = f"Pendiente 13w: {slope_w:.2f}%/sem | 4w: {ret4w:.1f}% | 52w: {ret52w:.1f}%"
        ax1.text(0.01, 0.02, subtitle, transform=ax1.transAxes, fontsize=9, va='bottom', ha='left')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.25)

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(macd.index, macd.values, label='MACD', color='tab:purple', linewidth=1.2)
        ax2.plot(macd_sig.index, macd_sig.values, label='Señal', color='tab:red', linewidth=1.0)
        colors = ['green' if v >= 0 else 'red' for v in macd_hist.values.ravel()]
        ax2.bar(macd_hist.index, macd_hist.values.ravel(),
        color=colors, alpha=0.4, width=5, linewidth=0.0)

        ax2.axhline(0, color='gray', linewidth=0.8)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.25)

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(rsi.index, rsi.values, color='tab:blue', linewidth=1.2, label='RSI(14)')
        ax3.axhline(70, color='gray', linestyle='--', linewidth=0.8)
        ax3.axhline(30, color='gray', linestyle='--', linewidth=0.8)
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.25)

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plots[ticker] = buf
        plt.close(fig)
    return plots

# Modelo simple
"""def simple_prediction(data):
    predictions = {}
    for ticker, df in data.items():
        df = df.reset_index()
        df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
        X = df[['Date_ordinal']]
        y = df['Close']
        model = LinearRegression().fit(X, y)
        next_day = pd.Timestamp.today().toordinal() + 1
        predictions[ticker] = model.predict([[next_day]])[0]
    return predictions
"""
# Función async para enviar mensajes por Telegram
async def send_telegram_analysis(data, plots):
    bot = telegram.Bot(token=bot_token)
    message = "📊 Análisis semanal de stocks:\n\n"
    
    # Enviar mensaje de texto
    await bot.send_message(chat_id=chat_id, text=message)
    
    # Enviar gráficos
    for ticker, plot_buf in plots.items():
        await bot.send_photo(chat_id=chat_id, photo=plot_buf)

# Ejecutar análisis
data = fetch_data(stocks)
plots = analyze_and_plot(data)
# preds = simple_prediction(data)

# Enviar por Telegram usando asyncio
try:
    asyncio.run(send_telegram_analysis(data, plots))
except Exception as e:
    print(f"Error al enviar el análisis por Telegram: {e}")
