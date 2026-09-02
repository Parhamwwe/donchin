"""
Donchian ATR Breakout - Telegram Signal Bot
============================================
این اسکریپت هر بار که اجرا می‌شه:
  1. از Binance آخرین کندل‌های بسته‌شده رو برای هر نماد/تایم‌فریم می‌گیره
  2. دقیقاً همون منطق اندیکاتور Pine Script (Donchian Breakout + فیلتر EMA/ADX/حجم) رو
     با پایتون محاسبه می‌کنه
  3. اگه سیگنال جدیدی (که قبلاً فرستاده نشده) پیدا کرد، با Entry/SL/TP به تلگرام می‌فرسته
  4. وضعیت (آخرین کندلی که پردازش شده) رو توی state.json ذخیره می‌کنه تا سیگنال تکراری نفرسته

این اسکریپت به‌صورت خودکار توسط GitHub Actions (کرون‌جاب رایگان) هر ساعت اجرا می‌شه.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════
# تنظیمات قابل تغییر — این بخش رو با نمادها/تایم‌فریم‌های خودتون عوض کنید
# ══════════════════════════════════════════════════════════════
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]   # نمادهای بایننس (بدون اسلش)
TIMEFRAMES = ["1h", "4h"]                      # تایم‌فریم‌های بایننس: 1h, 4h, 15m, ...

# پارامترهای استراتژی (دقیقاً مطابق نسخه‌ی Pine Script)
DONCHIAN_LEN = 20
USE_TREND_FILTER = True
TREND_EMA_LEN = 100
USE_ADX_FILTER = True
ADX_LEN = 14
ADX_THRESHOLD = 22
USE_VOLUME_FILTER = True
VOL_EMA_LEN = 20
ATR_LEN = 14
SL_ATR_MULT = 2.0
RR_TARGET = 2.0   # نسبت TP به SL (چون این سیگنال یه پیام ثابته، از TP ثابت استفاده می‌کنیم نه تریلینگ)

STATE_FILE = "state.json"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


# ══════════════════════════════════════════════════════════════
# توابع کمکی — اندیکاتورها
# ══════════════════════════════════════════════════════════════
def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """میانگین‌گیری وایلدر — همون چیزی که Pine Script برای ATR/ADX استفاده می‌کنه."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder_smooth(tr, period)


def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = compute_atr(df, period)
    plus_di = 100 * wilder_smooth(pd.Series(plus_dm, index=df.index), period) / atr
    minus_di = 100 * wilder_smooth(pd.Series(minus_dm, index=df.index), period) / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = wilder_smooth(dx, period)
    return adx


# ══════════════════════════════════════════════════════════════
# دریافت داده از بایننس
# ══════════════════════════════════════════════════════════════
def fetch_klines(symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["close_time"] = df["close_time"].astype(np.int64)

    # فقط کندل‌های کاملاً بسته‌شده رو نگه می‌داریم (کندل در حال شکل‌گیری رو حذف می‌کنیم)
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
# محاسبه‌ی سیگنال روی آخرین کندل بسته‌شده
# ══════════════════════════════════════════════════════════════
def compute_signal(df: pd.DataFrame):
    if len(df) < max(DONCHIAN_LEN, TREND_EMA_LEN, ADX_LEN, VOL_EMA_LEN) + 5:
        return None  # داده‌ی کافی نیست

    df = df.copy()
    df["donchian_high"] = df["high"].rolling(DONCHIAN_LEN).max().shift(1)
    df["donchian_low"] = df["low"].rolling(DONCHIAN_LEN).min().shift(1)
    df["trend_ema"] = df["close"].ewm(span=TREND_EMA_LEN, adjust=False).mean()
    df["atr"] = compute_atr(df, ATR_LEN)
    df["adx"] = compute_adx(df, ADX_LEN)
    df["vol_avg"] = df["volume"].rolling(VOL_EMA_LEN).mean()

    last = df.iloc[-1]

    breakout_long = last["close"] > last["donchian_high"]
    breakout_short = last["close"] < last["donchian_low"]

    trend_up_ok = (not USE_TREND_FILTER) or (last["close"] > last["trend_ema"])
    trend_down_ok = (not USE_TREND_FILTER) or (last["close"] < last["trend_ema"])

    adx_ok = (not USE_ADX_FILTER) or (last["adx"] > ADX_THRESHOLD)
    vol_ok = (not USE_VOLUME_FILTER) or (last["volume"] > last["vol_avg"])

    long_signal = breakout_long and trend_up_ok and adx_ok and vol_ok
    short_signal = breakout_short and trend_down_ok and adx_ok and vol_ok

    if not long_signal and not short_signal:
        return None

    direction = "LONG" if long_signal else "SHORT"
    entry = last["close"]
    atr_val = last["atr"]

    if direction == "LONG":
        sl = entry - atr_val * SL_ATR_MULT
        tp = entry + (entry - sl) * RR_TARGET
    else:
        sl = entry + atr_val * SL_ATR_MULT
        tp = entry - (sl - entry) * RR_TARGET

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "close_time": int(last["close_time"]),
        "adx": last["adx"],
    }


# ══════════════════════════════════════════════════════════════
# مدیریت وضعیت (جلوگیری از سیگنال تکراری)
# ══════════════════════════════════════════════════════════════
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ══════════════════════════════════════════════════════════════
# ارسال به تلگرام
# ══════════════════════════════════════════════════════════════
def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_BOT_TOKEN یا TELEGRAM_CHAT_ID تنظیم نشده — پیام فرستاده نشد.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }, timeout=20)
    if resp.status_code != 200:
        print(f"❌ خطا در ارسال تلگرام: {resp.status_code} {resp.text}")
    else:
        print("✅ پیام تلگرام ارسال شد.")


def format_signal_message(symbol: str, timeframe: str, sig: dict) -> str:
    emoji = "🟢" if sig["direction"] == "LONG" else "🔴"
    return (
        f"{emoji} <b>{sig['direction']} SIGNAL</b> — {symbol} ({timeframe})\n\n"
        f"Entry: <code>{sig['entry']:.4f}</code>\n"
        f"SL: <code>{sig['sl']:.4f}</code>\n"
        f"TP: <code>{sig['tp']:.4f}</code>\n"
        f"ADX: {sig['adx']:.1f}\n\n"
        f"⚠️ این یه سیگنال خودکاره، نه توصیه‌ی مالی. حتماً ریسک هر معامله رو خودتون مدیریت کنید."
    )


# ══════════════════════════════════════════════════════════════
# اجرای اصلی
# ══════════════════════════════════════════════════════════════
def main():
    state = load_state()
    updated = False

    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            key = f"{symbol}_{tf}"
            try:
                df = fetch_klines(symbol, tf)
                sig = compute_signal(df)
            except Exception as e:
                print(f"❌ خطا برای {key}: {e}")
                continue

            if sig is None:
                continue

            last_processed = state.get(key)
            if last_processed == sig["close_time"]:
                # این کندل قبلاً پردازش و سیگنالش (اگه بوده) فرستاده شده
                continue

            # سیگنال جدید پیدا شد
            msg = format_signal_message(symbol, tf, sig)
            send_telegram_message(msg)

            state[key] = sig["close_time"]
            updated = True

    if updated:
        save_state(state)
    else:
        print("سیگنال جدیدی پیدا نشد.")


if __name__ == "__main__":
    main()
