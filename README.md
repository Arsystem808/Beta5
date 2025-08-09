
# CapIntel AI (GPT-5 • STOOQ FIX)

## Главное
- Если Yahoo даёт пусто, подключается **Stooq** (через pandas-datareader) — берём дневные котировки за 5 лет.
- Остальной функционал: авто-горизонт, дальние цели, скрытая методика, стиль «как у тебя», GPT‑5 с `max_completion_tokens`.

## Запуск
1) `pip install -r requirements.txt`
2) `.streamlit/secrets.toml`:
```
OPENAI_API_KEY = "sk-..."
```
3) `streamlit run app.py`
