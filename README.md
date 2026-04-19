<<<<<<< HEAD
# FactLens - AI Fake News Detector

[![Render Deploy](https://render.com/images/deploy-to-render.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USERNAME/FactLens)

## Features
- ML model (95.76% accuracy) detects fake news via TF-IDF + Logistic Regression
- Live news integration (NewsAPI + Google RSS)
- Related articles with similarity scores (20%+ threshold)
- Full news browsing with search/pagination

## Local Setup
```bash
py -3 -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python model.py  # Train model (if needed)
python app.py
```
Visit http://127.0.0.1:5000

## Deploy to Render
1. Fork/Clone repo
2. New Web Service → Connect GitHub repo
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app:app`
5. Add NEWS_API_KEY env var

## API Endpoints
- POST /predict - Fact-check text
- GET /news - Latest headlines
- GET /news_page - Full news browser

Built with Flask, scikit-learn. Model trained on fact-checking datasets.

=======
# FactLens
>>>>>>> 19d170517a5d18887a7612895bcd259865e5794e
