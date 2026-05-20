# FactLens - AI Fake News Detector

[![Render Deploy](https://render.com/images/deploy-to-render.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/FactLens)

## Features
- ML model (90.72% accuracy) detects fake news via TF-IDF + Logistic Regression
- Live news integration (NewsAPI + Google RSS)
- Related articles with similarity scores (20%+ threshold)
- Full news browsing with search/pagination

## Local Setup
```bash
py -3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python model.py  # Train model (if needed)
python app.py
```
Visit http://127.0.0.1:5000

## Deploy to Render
1. Fork/Clone repo
2. New Web Service -> Connect GitHub repo
3. Build: `pip install -r requirements.txt`
4. Start: `FACTLENS_USE_BERT=0 gunicorn --workers 1 --timeout 120 app:app`
5. Add NEWS_API_KEY env var

### Render model mode
Render free/small instances can kill the worker when PyTorch/BERT loads. The
default Render start command sets `FACTLENS_USE_BERT=0`, so deployment uses a
lightweight hosted fallback encoder and still returns predictions instead of
crashing. Local runs still use BERT by default.

To run full BERT on a larger Render instance, set `FACTLENS_USE_BERT=1`, use one
worker, and cache the model during build:

```bash
pip install -r requirements.txt && python scripts/cache_bert.py
```

## API Endpoints
- POST /predict - Fact-check text
- GET /news - Latest headlines
- GET /news_page - Full news browser

Built with Flask, scikit-learn. Model trained on fact-checking datasets.
