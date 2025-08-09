
import streamlit as st
from openai import OpenAI
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, random

st.set_page_config(page_title="CapIntel AI (GPT-5 • STOOQ FIX)", layout="centered")
st.title("CapIntel AI (GPT-5 • STOOQ FIX)")

# --------- Diagnostics ---------
with st.sidebar:
    st.header("Диагностика")
    key_ok = ("OPENAI_API_KEY" in st.secrets) or bool(os.getenv("OPENAI_API_KEY"))
    st.write("Ключ:", "ОК" if key_ok else "нет")
    st.write("Streamlit:", st.__version__)
    try:
        import openai as _oa
        st.write("openai:", _oa.__version__)
    except Exception as _e:
        st.write("openai import error:", str(_e)[:120])
    try:
        yf.Ticker("QQQ").history(period="1d")
        st.write("Yahoo Finance: ОК")
    except Exception as e:
        st.write("Yahoo Finance: ошибка", str(e)[:120])

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Нет OPENAI_API_KEY. Добавь в .streamlit/secrets.toml или переменные окружения.")
    return OpenAI(api_key=api_key)

ticker = st.text_input("Введите тикер (например, QQQ, AMD, SNOW):").upper().strip()
horizon_choice = st.selectbox(
    "Горизонт:",
    ["Авто (ИИ выберет)", "Трейд (1-5 дней)", "Среднесрок (1-4 недели)", "Долгосрок (1-6 месяцев)"]
)
run = st.button("Проанализировать")

def round2(x):
    return None if x is None else round(float(x), 2)

# --------- Stooq fallback ---------
def _stooq_download(ticker: str, years: int = 5) -> pd.DataFrame:
    try:
        from pandas_datareader import data as pdr
    except Exception:
        return pd.DataFrame()
    sym = f"{ticker}.US"
    start = datetime.utcnow() - timedelta(days=365*years)
    try:
        df = pdr.DataReader(sym, "stooq", start=start)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.sort_index()
            # stooq columns: Open, High, Low, Close, Volume
            need = {"Open","High","Low","Close"}
            for m in (need - set(df.columns)):
                df[m] = np.nan
            return df[["Open","High","Low","Close"]].dropna(how="all")
    except Exception:
        pass
    return pd.DataFrame()

# --------- Robust Yahoo loader with Stooq fallback ---------
@st.cache_data(ttl=300)
def _history_resilient(ticker: str, period: str, interval: str) -> pd.DataFrame:
    def ok(df):
        return isinstance(df, pd.DataFrame) and not df.empty
    def normalize(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        need = {"Open","High","Low","Close"}
        for m in (need - set(df.columns)):
            df[m] = np.nan
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[["Open","High","Low","Close"]].dropna(how="all")

    attempts = []
    # 1) Yahoo как просили
    attempts.append(("yh", dict(period=period, interval=interval)))
    # 2) Интрадей альтернативы
    intraday_map = {"1h":["60m","90m","30m"], "60m":["90m","30m"], "90m":["60m","30m"]}
    for iv in intraday_map.get(interval, []):
        attempts.append(("yh", dict(period=period, interval=iv)))
    # 3) Дневки с расширением периода
    for p in ["30d","3mo","6mo","1y","2y","5y","max"]:
        attempts.append(("yh", dict(period=p, interval="1d")))
    # 4) STOOQ fallback (5 лет дневок)
    attempts.append(("stooq", dict(years=5)))

    for kind, kw in attempts:
        try:
            if kind == "yh":
                df = yf.Ticker(ticker).history(auto_adjust=False, actions=False, threads=False, **kw)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return normalize(df)
            else:
                df = _stooq_download(ticker, **kw)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    return normalize(df)
        except Exception:
            continue
    return pd.DataFrame()

def atr(df: pd.DataFrame, period: int = 14) -> float:
    hi = df["High"]; lo = df["Low"]; cl = df["Close"]
    tr = pd.concat([(hi - lo), (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    v = tr.rolling(period).mean().dropna()
    return float(v.iloc[-1]) if len(v)>0 else float(tr.mean())

def fib_pivots_from_prev(H, L, C):
    P = (H + L + C) / 3.0
    R = (H - L)
    R1 = P + 0.382 * R
    R2 = P + 0.618 * R
    R3 = P + 1.000 * R
    S1 = P - 0.382 * R
    S2 = P - 0.618 * R
    S3 = P - 1.000 * R
    return P, R1, R2, R3, S1, S2, S3

def fib_pivot_extensions(P, R):
    R4 = P + 1.272 * R
    R5 = P + 1.618 * R
    S4 = P - 1.272 * R
    S5 = P - 1.618 * R
    return R4, R5, S4, S5

def prev_period_hlc(ticker: str, period: str):
    base = _history_resilient(ticker, period="5y", interval="1d")
    if base is None or base.empty:
        raise ValueError("Не удалось получить базовые данные для расчёта уровней.")
    base = base.dropna()
    if period == "day":
        if len(base) < 2: raise ValueError("Недостаточно дневных данных.")
        prev = base.iloc[-2]
    elif period == "week":
        w = base.resample("W").agg({"High":"max","Low":"min","Close":"last"}).dropna()
        if len(w) < 2: raise ValueError("Недостаточно недельных данных.")
        prev = w.iloc[-2]
    elif period == "month":
        m = base.resample("M").agg({"High":"max","Low":"min","Close":"last"}).dropna()
        if len(m) < 2: raise ValueError("Недостаточно месячных данных.")
        prev = m.iloc[-2]
    elif period == "year":
        y = base.resample("Y").agg({"High":"max","Low":"min","Close":"last"}).dropna()
        if len(y) < 2: raise ValueError("Недостаточно годовых данных.")
        prev = y.iloc[-2]
    else:
        raise ValueError("Unknown period")
    return float(prev["High"]), float(prev["Low"]), float(prev["Close"])

def get_hist_for_horizon(ticker: str, horizon: str) -> pd.DataFrame:
    if horizon == "Трейд (1-5 дней)":
        df = _history_resilient(ticker, period="10d", interval="1h")
        if df.empty:
            df = _history_resilient(ticker, period="30d", interval="1d")
        if df.empty:
            df = _history_resilient(ticker, period="3mo", interval="1d")
        return df

    if horizon == "Среднесрок (1-4 недели)":
        df = _history_resilient(ticker, period="1mo", interval="1d")
        if df.empty:
            df = _history_resilient(ticker, period="3mo", interval="1d")
        if df.empty:
            df = _history_resilient(ticker, period="6mo", interval="1d")
        return df

    # Долгосрок
    df = _history_resilient(ticker, period="6mo", interval="1d")
    if df.empty:
        df = _history_resilient(ticker, period="1y", interval="1d")
    if df.empty:
        df = _history_resilient(ticker, period="5y", interval="1d")
    return df

def classify_zone(price, p, r1, r2, r3, s1, s2, s3):
    if price >= r3: return "ultra_upper"
    if price >= r2: return "upper"
    if price >= r1: return "mid_upper"
    if price >= p:  return "around_p"
    if price >= s1: return "mid_lower"
    if price >= s2: return "lower"
    return "deep_lower"

def heikin_ashi_close(df: pd.DataFrame) -> pd.Series:
    return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0

def macd_hist(close: pd.Series) -> pd.Series:
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal

def sign_streak(series: pd.Series):
    s = np.sign(series.dropna())
    if len(s)==0: return 0
    last = s.iloc[-1]
    cnt=0
    for v in reversed(s.values):
        if v==last and v!=0: cnt+=1
        else: break
    return int(cnt * (1 if last>0 else -1))

def horizon_params(h):
    if h == "Трейд (1-5 дней)":
        return {"atr_k": 1.0, "min_pct": 0.012}
    if h == "Среднесрок (1-4 недели)":
        return {"atr_k": 1.7, "min_pct": 0.06}
    return {"atr_k": 2.3, "min_pct": 0.12}

def fib_pivots_from_prev(H, L, C):
    P = (H + L + C) / 3.0
    R = (H - L)
    R1 = P + 0.382 * R
    R2 = P + 0.618 * R
    R3 = P + 1.000 * R
    S1 = P - 0.382 * R
    S2 = P - 0.618 * R
    S3 = P - 1.000 * R
    return P, R1, R2, R3, S1, S2, S3

def fib_pivot_extensions(P, R):
    R4 = P + 1.272 * R
    R5 = P + 1.618 * R
    S4 = P - 1.272 * R
    S5 = P - 1.618 * R
    return R4, R5, S4, S5

def pick_targets(direction, piv_base, horizon, price_now, piv_year=None, piv_month=None):
    P,R1,R2,R3,S1,S2,S3 = piv_base
    width = abs((R1 - P)) + abs((P - S1))
    if width == 0:
        width = abs(R3 - S3) if (R3 and S3) else max(1.0, P*0.05)
    width = abs(width) / 2.0
    extR4, extR5, extS4, extS5 = fib_pivot_extensions(P, width)

    base_long  = [x for x in [R1,R2,R3, extR4, extR5] if x is not None]
    base_short = [x for x in [S1,S2,S3, extS4, extS5] if x is not None]

    ctx = []
    for piv in (piv_year, piv_month):
        if piv:
            P2,R1b,R2b,R3b,S1b,S2b,S3b = piv
            ctx += [R1b,R2b,R3b,S1b,S2b,S3b]

    if direction == "LONG":
        cands = [x for x in base_long + ctx if x > price_now]
        cands = sorted(set([float(x) for x in cands]))
    else:
        cands = [x for x in base_short + ctx if x < price_now]
        cands = sorted(set([float(x) for x in cands]), reverse=True)

    need = horizon_params(horizon)["min_pct"]
    use = [x for x in cands if abs(x - price_now) / max(price_now, 1e-6) >= need]
    if not use: use = cands
    if not use: return None, None
    if len(use) == 1: return use[0], use[0]
    return use[0], use[1]

def make_plan(price_now, horizon, piv_main, df_for_atr, piv_ctx_year=None, piv_ctx_month=None):
    P,R1,R2,R3,S1,S2,S3 = piv_main
    zone = classify_zone(price_now, P,R1,R2,R3,S1,S2,S3)

    ha = heikin_ashi_close(df_for_atr).diff()
    mh = macd_hist(df_for_atr["Close"])
    ha_len = sign_streak(ha)
    macd_len = sign_streak(mh)
    overextended = (
        (horizon=="Долгосрок (1-6 месяцев)" and ha_len>=5 and macd_len>=5 and price_now>=R2) or
        (horizon=="Среднесрок (1-4 недели)" and ha_len>=4 and macd_len>=4 and price_now>=R1)
    )

    pars = horizon_params(horizon)
    atr_k = pars["atr_k"]
    curr_atr = atr(df_for_atr)

    if overextended or zone in ("upper","ultra_upper"):
        base = "SHORT"
    elif zone in ("deep_lower","lower","mid_lower"):
        base = "LONG"
    else:
        base = "WAIT"

    entry=tp1=tp2=sl=None
    alt_action=alt_entry=alt_tp1=alt_tp2=alt_sl=None

    if base=="LONG":
        entry = max(S1, min(P, price_now))
        tp1,tp2 = pick_targets("LONG",  piv_main, horizon, price_now, piv_ctx_year, piv_ctx_month)
        sl = entry - atr_k*curr_atr
        alt_action="SHORT"; alt_entry=R2 if R2 else price_now
        alt_tp1,alt_tp2 = pick_targets("SHORT", piv_main, horizon, price_now, piv_ctx_year, piv_ctx_month)
        alt_sl = alt_entry + atr_k*curr_atr

    elif base=="SHORT":
        entry = min(R2 if R2 else price_now, R3 if R3 else price_now)
        tp1,tp2 = pick_targets("SHORT", piv_main, horizon, price_now, piv_ctx_year, piv_ctx_month)
        sl = entry + atr_k*curr_atr
        alt_action="LONG"; alt_entry=max(S1, min(P, price_now))
        alt_tp1,alt_tp2 = pick_targets("LONG",  piv_main, horizon, price_now, piv_ctx_year, piv_ctx_month)
        alt_sl = alt_entry - atr_k*curr_atr

    else:  # WAIT
        if price_now>=P:
            alt_action="SHORT"; alt_entry=R2 if R2 else price_now
            alt_tp1,alt_tp2 = pick_targets("SHORT", piv_main, horizon, price_now, piv_ctx_year, piv_ctx_month)
            alt_sl = alt_entry + atr_k*curr_atr
        else:
            alt_action="LONG"; alt_entry=max(S1, min(P, price_now))
            alt_tp1,alt_tp2 = pick_targets("LONG",  piv_main, horizon, price_now, piv_ctx_year, piv_ctx_month)
            alt_sl = alt_entry - atr_k*curr_atr

    return {
        "zone": zone,
        "base_action": base,
        "entry": round2(entry), "tp1": round2(tp1), "tp2": round2(tp2), "sl": round2(sl),
        "alt_action": alt_action, "alt_entry": round2(alt_entry),
        "alt_tp1": round2(alt_tp1), "alt_tp2": round2(alt_tp2), "alt_sl": round2(alt_sl)
    }

def forecast_sentence(price, piv_main):
    P,R1,R2,R3,S1,S2,S3 = piv_main
    if price >= P:
        lo = round2((P+S1)/2)
        return f"если начнётся сброс, логично увидеть возврат к {lo}; при сохранении давления вверх фокус сместится к {round2(R1)}, далее к {round2(R2)}."
    else:
        hi = round2((P+R1)/2)
        return f"если удастся удержаться и подтянуться выше, дорога открыта к {hi}; в случае слабости рискуем уйти к {round2(S1)}."

def auto_choose_horizon(ticker: str):
    try:
        H_y,L_y,C_y = prev_period_hlc(ticker, "year")
        piv_y = fib_pivots_from_prev(H_y,L_y,C_y)
    except Exception:
        piv_y = None
    try:
        H_m,L_m,C_m = prev_period_hlc(ticker, "month")
        piv_m = fib_pivots_from_prev(H_m,L_m,C_m)
    except Exception:
        piv_m = None

    price_hist = _history_resilient(ticker, period="5d", interval="1d")
    price = None if price_hist is None or price_hist.empty else float(price_hist["Close"].iloc[-1])
    if price is None:
        return "Среднесрок (1-4 недели)", "цена недоступна → дефолт среднесрок"

    def near_extremes(price, piv):
        P,R1,R2,R3,S1,S2,S3 = piv
        span = max(1e-6, R3 - S3)
        upper = 1.0 - (R2 - price)/max(1e-6, span) if price<=R3 else 1.0
        lower = 1.0 - (price - S2)/max(1e-6, span) if price>=S3 else 1.0
        return max(0.0, upper), max(0.0, lower)

    if piv_y:
        u,l = near_extremes(price, piv_y)
        if u>0.85 or l>0.85:
            return "Долгосрок (1-6 месяцев)", f"близко к крайним годовым ({u:.2f}/{l:.2f})"

    if piv_m:
        u,l = near_extremes(price, piv_m)
        if u>0.75 or l>0.75:
            return "Среднесрок (1-4 недели)", f"близко к крайним месячным ({u:.2f}/{l:.2f})"

    return "Трейд (1-5 дней)", "далеко от крайних уровней → краткосрок"

if run and ticker:
    with st.spinner("Загружаем данные и формируем взгляд..."):
        try:
            if horizon_choice == "Авто (ИИ выберет)":
                horizon, auto_reason = auto_choose_horizon(ticker)
            else:
                horizon, auto_reason = horizon_choice, ""

            df = get_hist_for_horizon(ticker, horizon)
            if df is None or df.empty:
                st.error("Нет данных по тикеру после всех попыток. Проверь тикер и попробуй ещё раз."); st.stop()

            st.caption(f"rows={len(df)}; first={df.index.min() if not df.empty else '—'}; last={df.index.max() if not df.empty else '—'}; cols={list(df.columns)}")
            if auto_reason:
                st.info(f"Автовыбор горизонта: **{horizon}** ({auto_reason})")
            else:
                st.caption(f"Горизонт выбран вручную: {horizon}")

            price_now = float(df["Close"].iloc[-1])

            if horizon == "Трейд (1-5 дней)":
                H,L,C = prev_period_hlc(ticker, "week")
            elif horizon == "Среднесрок (1-4 недели)":
                H,L,C = prev_period_hlc(ticker, "month")
            else:
                H,L,C = prev_period_hlc(ticker, "year")

            piv_main = fib_pivots_from_prev(H,L,C)

            piv_ctx_year = piv_ctx_month = None
            try:
                Hy,Ly,Cy = prev_period_hlc(ticker, "year")
                piv_ctx_year = fib_pivots_from_prev(Hy,Ly,Cy)
            except: pass
            try:
                Hm,Lm,Cm = prev_period_hlc(ticker, "month")
                piv_ctx_month = fib_pivots_from_prev(Hm,Lm,Cm)
            except: pass

            plan = make_plan(price_now, horizon, piv_main, df_for_atr=df,
                             piv_ctx_year=piv_ctx_year, piv_ctx_month=piv_ctx_month)

            snapshot = {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                "ticker": ticker, "horizon": horizon, "price": round2(price_now),
                "P": round2(piv_main[0]), "R1": round2(piv_main[1]), "R2": round2(piv_main[2]), "R3": round2(piv_main[3]),
                "S1": round2(piv_main[4]), "S2": round2(piv_main[5]), "S3": round2(piv_main[6]),
                **plan
            }
            snapshot["forecast"] = forecast_sentence(price_now, piv_main)
        except Exception as e:
            st.error(f"Ошибка при загрузке истории/уровней: {e}"); st.stop()

    try:
        client = get_openai_client()

        def v(x): return "" if x in (None, "None") else str(x)
        main_label = "WAIT" if snapshot["base_action"]=="WAIT" else ("Лонг" if snapshot["base_action"]=="LONG" else "Шорт")
        alt_label  = "" if snapshot["alt_action"] in (None, "None") else ("лонг" if snapshot["alt_action"]=="LONG" else "шорт")

        wait_hint = ""
        if snapshot["base_action"]=="WAIT":
            P = float(snapshot["P"]) if snapshot["P"] is not None else None
            S1 = float(snapshot["S1"]) if snapshot["S1"] is not None else None
            R1 = float(snapshot["R1"]) if snapshot["R1"] is not None else None
            if P and S1 and snapshot["price"]>=P:
                lo = round((P+S1)/2,2); hi = round(P,2)
                wait_hint = f"{lo}–{hi}"
            elif P and R1:
                lo = round(P,2); hi = round((P+R1)/2,2)
                wait_hint = f"{lo}–{hi}"

        style_variants = [
            "Видится мне, что рынок начинает вязнуть — импульс гаснет, тени сверху толстеют.",
            "По моему ощущению рынка, покупатели устали: ходы короче, дыхание сбито.",
            "Сверху нащупывается потолок — цена как будто ищет повод остыть.",
            "Движение выдыхается, ощущение такое, что рынок просит паузу перед следующим шагом."
        ]
        closing_variants = [
            "Действуем по плану: есть подтверждение — работаем, нет — отходим в сторону.",
            "Главное — дисциплина: сигнал есть — входим, нарушился — выходим без сожалений.",
            "Сценарий простой: подтверждение → вход; ломка → пауза и ждём следующую точку."
        ]
        open_line  = random.choice(style_variants)
        close_line = random.choice(closing_variants)

        system_prompt = (
            "Пиши как опытный трейдер Арсен: живо, уверенно, без раскрытия методик. "
            "Не упоминай индикаторы, уровни и их названия — только цены и действия. "
            "Не используй эмодзи. Избегай штампов и повторов. Иногда меняй порядок: действие→причина или наоборот. "
            "Строгий формат:\n"
            "Результат анализа: {TICKER} — {HORIZON}\n\n"
            "2–3 фразы вступления (импульс/усталость). Допускай: 'видится мне...', 'по моему ощущению рынка...'.\n"
            "—\n\n"
            "Рекомендация: {MAIN}\n\n"
            "→ 1–3 строки: что делать и почему. Если WAIT — диапазон X–Y, где ждать вход.\n"
            "—\n\n"
            "Альтернатива: {ALT}\n\n"
            "→ Противоположная сторона: вход, 2 цели, защита. Кому подходит.\n"
            "—\n\n"
            "Что дальше: краткий прогноз после коррекции/пробоя (конкретные ориентиры). "
            "В конце — короткий вывод. Цифры менять запрещено."
        )

        user_prompt = f"""
TICKER={snapshot['ticker']}; HORIZON={snapshot['horizon']}; PRICE={snapshot['price']}

Результат анализа: {snapshot['ticker']} — {snapshot['horizon']}

{open_line}

—

Рекомендация: {main_label}

{"→ Вход сейчас не даёт преимущества — ждём область " + wait_hint + " для новой попытки." if snapshot['base_action']=="WAIT" else ""}
{"→ Лонг: набор от " + v(snapshot['entry']) + ", ориентиры " + v(snapshot['tp1']) + " и " + v(snapshot['tp2']) + ", защита " + v(snapshot['sl']) + "." if snapshot['base_action']=="LONG" else ""}
{"→ Шорт: работа от " + v(snapshot['entry']) + ", цели " + v(snapshot['tp1']) + " и " + v(snapshot['tp2']) + ", защита выше " + v(snapshot['sl']) + "." if snapshot['base_action']=="SHORT" else ""}

—

{("Альтернатива: " + alt_label + "\n\n→ Для тех, кто умеет работать с откатами и стопом: вход около " + v(snapshot['alt_entry']) + ", цели " + v(snapshot['alt_tp1']) + " и " + v(snapshot['alt_tp2']) + ", защита " + v(snapshot['alt_sl']) + ".") if snapshot['alt_action'] not in (None, "None") else ""}

—

Что дальше: {snapshot['forecast']}

Итог: {close_line}
"""
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            max_completion_tokens=900
        )
        text = resp.choices[0].message.content.strip()

        st.markdown(f"### {snapshot['ticker']} — текущая цена: ${round(snapshot['price'],2)}")
        st.markdown("### Результат:")
        st.write(text)

        with st.expander("Внутренние уровни (для владельца)"):
            st.json({k: snapshot[k] for k in ["P","R1","R2","R3","S1","S2","S3","zone","base_action"]})

    except Exception as e:
        st.error(f"Ошибка генерации текста: {e}")
