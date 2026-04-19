from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import joblib
import requests
from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

from model import MODEL_PATH, TFIDF_PATH, clean_text


BASE_DIR = Path(__file__).resolve().parent


def load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

NEWS_CACHE_SECONDS = int(os.getenv("NEWS_CACHE_SECONDS", "900"))
NEWS_API_URL = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
NEWS_DEFAULT_QUERY = os.getenv(
    "NEWS_DEFAULT_QUERY",
    "world OR breaking OR politics OR economy OR technology OR health OR climate",
)
DEFAULT_NEWS_IMAGE = "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=900&q=80"
TOPIC_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "been",
    "before",
    "being",
    "between",
    "could",
    "from",
    "have",
    "into",
    "latest",
    "more",
    "news",
    "over",
    "said",
    "says",
    "that",
    "their",
    "there",
    "this",
    "today",
    "with",
    "would",
}
MIN_ARTICLE_WORDS = 3
MIN_PREDICTION_WORDS = 10
FAKE_CONFIDENCE_THRESHOLD = 95.0
REAL_CONFIDENCE_THRESHOLD = 60.0
EVIDENCE_SUPPORT_THRESHOLD = 2
EVIDENCE_SIMILARITY_THRESHOLD = 0.75
HIGH_EVIDENCE_SIMILARITY_THRESHOLD = 0.99
TRUSTED_SOURCES = {
    "ABC News",
    "ABC News (AU)",
    "Al Jazeera English",
    "Associated Press",
    "BBC News",
    "Bloomberg",
    "CBS News",
    "CNN",
    "Financial Times",
    "Fox News",
    "National Institutes of Health",
    "NPR",
    "Reuters",
    "Science Daily",
    "The Associated Press",
    "The Guardian",
    "The Hindu",
    "The Indian Express",
    "The Irish Times",
    "The Times of India",
    "The Wall Street Journal",
    "The Washington Post",
}

_model = None
_tfidf = None
_news_cache: dict[str, dict[str, Any]] = {}


def load_artifacts() -> None:
    global _model, _tfidf
    if not Path(MODEL_PATH).exists() or not Path(TFIDF_PATH).exists():
        raise FileNotFoundError("model.pkl and tfidf.pkl are missing. Run: python model.py")
    _model = joblib.load(MODEL_PATH)
    _tfidf = joblib.load(TFIDF_PATH)


@lru_cache(maxsize=512)
def predict_article(raw_text: str) -> dict[str, Any]:
    if _model is None or _tfidf is None:
        load_artifacts()

    cleaned = clean_text(raw_text)
    word_count = len(cleaned.split())
    features = _tfidf.transform([cleaned])
    probabilities = _model.predict_proba(features)[0]
    probability_index = int(probabilities.argmax())
    predicted_label = int(_model.classes_[probability_index])
    confidence = float(probabilities[probability_index])
    confidence_percent = round(confidence * 100, 2)
    model_prediction = "Real" if predicted_label == 1 else "Fake"

    return {
        "prediction": model_prediction,
        "modelConfidence": confidence_percent,
        "confidence": confidence_percent,
        "label": predicted_label,
        "modelPrediction": model_prediction,
        "wordCount": word_count,
    }


def extract_topic(text: str, max_words: int = 5) -> str:
    text = re.sub(r"\b(bbc|cnn|reuters|ap|npr|abc|cbs|fox)\s+(reports?|news|says?)\b", " ", text, flags=re.I)
    proper_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
    if proper_phrases:
        return proper_phrases[0]

    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    useful_words = [word for word in words if word not in TOPIC_STOPWORDS]
    return " ".join(useful_words[:max_words])


def extract_evidence_query(text: str, max_words: int = 12) -> str:
    cleaned_text = re.sub(r"\b(bbc|cnn|reuters|ap|npr|abc|cbs|fox)\s+(reports?|news|says?)\b", " ", text, flags=re.I)
    proper_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", cleaned_text)
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", cleaned_text.lower())
    useful_words = []
    for word in words:
        if word in TOPIC_STOPWORDS or word in useful_words:
            continue
        useful_words.append(word)

    query_parts = []
    if proper_phrases:
        query_parts.append(proper_phrases[0])
        proper_words = set(proper_phrases[0].lower().split())
        useful_words = [word for word in useful_words if word not in proper_words]
    query_parts.extend(useful_words[: max(0, max_words - len(" ".join(query_parts).split()))])
    return " ".join(query_parts).strip()


def has_enough_article_text(title: str, description: str) -> bool:
    combined = f"{title} {description}"
    title_words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", title)
    if len(title_words) < 4:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", combined)
    visible_chars = re.sub(r"\s+", "", combined)
    if not visible_chars:
        return False
    ascii_letters = re.findall(r"[A-Za-z]", combined)
    english_ratio = len(ascii_letters) / len(visible_chars)
    return len(words) >= MIN_ARTICLE_WORDS and english_ratio >= 0.40


def score_supporting_articles(
    raw_text: str,
    articles: list[dict[str, str]],
    exclude_url: str = "",
) -> list[dict[str, Any]]:
    if _tfidf is None:
        load_artifacts()
    if not articles:
        return []

    input_title = clean_text(raw_text.splitlines()[0] if raw_text.splitlines() else raw_text)
    filtered_articles = []
    for article in articles:
        if exclude_url and article.get("url") == exclude_url:
            continue
        if clean_text(article.get("title", "")) == input_title:
            continue
        filtered_articles.append(article)
    if not filtered_articles:
        return []

    article_texts = [
        f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        for article in filtered_articles
    ]
    vectors = _tfidf.transform([clean_text(raw_text), *[clean_text(text) for text in article_texts]])
    tfidf_similarities = cosine_similarity(vectors[0], vectors[1:]).ravel()

    raw_title = raw_text.splitlines()[0] if raw_text.splitlines() else raw_text
    clean_raw_title = clean_text(raw_title)
    clean_raw_text = clean_text(raw_text)

    scored_articles = []
    LOW_SIM_THRESHOLD = 20.0
    for article, tfidf_score in zip(filtered_articles, tfidf_similarities):
        article_title = clean_text(article.get("title", ""))
        article_text = clean_text(f"{article.get('title', '')} {article.get('description', '')}")
        title_score = SequenceMatcher(None, clean_raw_title, article_title).ratio()
        text_score = SequenceMatcher(None, clean_raw_text, article_text).ratio()
        score = max(float(tfidf_score), title_score, text_score)
        item = dict(article)
        item["similarity"] = round(score * 100, 2)
        if score >= EVIDENCE_SIMILARITY_THRESHOLD:
            item["matchType"] = "Supporting"
        elif score >= LOW_SIM_THRESHOLD:
            item["matchType"] = "Possible"
        else:
            continue
        scored_articles.append(item)

        app.logger.info(f"Showing {len(scored_articles)} related articles with min 20% similarity")
    return sorted(scored_articles, key=lambda article: article["similarity"], reverse=True)


def build_verdict(raw_text: str, source: str = "", url: str = "") -> dict[str, Any]:
    model_result = predict_article(raw_text)
    topic_query = extract_topic(raw_text)
    evidence_query = extract_evidence_query(raw_text)
    supporting_articles: list[dict[str, str]] = []
    evidence_error = None

    if evidence_query:
        app.logger.debug(f"Evidence query: {evidence_query}")
        candidate_articles, evidence_error = fetch_evidence_candidates(query=evidence_query, page_size=12)
        app.logger.debug(f"Candidate articles: {len(candidate_articles)}, error: {evidence_error}")
        supporting_articles = score_supporting_articles(raw_text, candidate_articles, exclude_url=url)

    trusted_support_count = sum(1 for article in supporting_articles if article.get("source") in TRUSTED_SOURCES)
    support_count = len(supporting_articles)
    best_similarity = max((float(article.get("similarity", 0)) for article in supporting_articles), default=0.0)
    exact_match = best_similarity >= 100.0

    prediction = model_result["modelPrediction"]
    confidence = float(model_result["modelConfidence"])
    word_count = int(model_result["wordCount"])

    if exact_match:
        final_prediction = "Real"
        verdict = "Exact Match Found"
        reason = "Identical article found in news sources."
    elif prediction == "Real" and confidence >= 95.0:
        final_prediction = "Real"
        verdict = "High Confidence Real"
        reason = "ML model predicts real with very high confidence."
    elif prediction == "Real":
        final_prediction = "Likely Real"
        verdict = "Model Predicts Real"
        reason = "ML model predicts real, but no exact match found."
    else:
        final_prediction = "Fake"
        verdict = "Model Predicts Fake"
        reason = "ML model predicts fake."

    return {
        **model_result,
        "prediction": final_prediction,
        "verdict": verdict,
        "reason": reason,
        "topic": topic_query,
        "evidenceQuery": evidence_query,
        "supportCount": support_count,
        "trustedSupportCount": trusted_support_count,
        "bestSimilarity": round(best_similarity, 2),
        "source": source,
        "url": url,
        "supportingArticles": supporting_articles[:3],
        "evidenceError": evidence_error,
    }


def parse_google_news_date(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError):
        return ""


def clean_google_title(title: str, source: str) -> str:
    suffix = f" - {source}"
    if source and title.endswith(suffix):
        return title[: -len(suffix)].strip()
    return title


def fetch_google_news_rss(query: str, page_size: int) -> tuple[list[dict[str, str]], str | None]:
    app.logger.debug(f"Fetching Google RSS for query: {query}")
    rss_url = f"{GOOGLE_NEWS_RSS_URL}?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(rss_url, timeout=8, headers={"User-Agent": "FactLens/1.0"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError) as e:
        app.logger.warning(f"Google RSS fetch failed: {e}")
        return [], "Google News RSS could not be loaded right now."

    payload = []
    for item in root.findall("./channel/item"):
        raw_title = item.findtext("title", default="Untitled story")
        source_node = item.find("source")
        source = source_node.text.strip() if source_node is not None and source_node.text else "Google News"
        title = clean_google_title(raw_title.strip(), source)
        description = re.sub(r"<[^>]+>", " ", item.findtext("description", default=""))
        description = re.sub(r"\s+", " ", description).strip()
        link = item.findtext("link", default="")

        if not has_enough_article_text(title, description):
            continue

        payload.append(
            {
                "title": title,
                "description": description,
                "content": description or title,
                "source": source,
                "image": DEFAULT_NEWS_IMAGE,
                "publishedAt": parse_google_news_date(item.findtext("pubDate", default="")),
                "url": link,
                "provider": "Google News RSS",
            }
        )
        if len(payload) >= page_size:
            break

    return payload, None


def dedupe_articles(articles: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique_articles = []
    for article in articles:
        key = article.get("url") or f"{article.get('source', '')}:{clean_text(article.get('title', ''))}"
        if key in seen:
            continue
        seen.add(key)
        unique_articles.append(article)
    return unique_articles


def fetch_evidence_candidates(query: str, page_size: int = 12) -> tuple[list[dict[str, str]], str | None]:
    primary_articles, primary_error = fetch_latest_news(query=query, page_size=page_size)
    rss_articles, rss_error = fetch_google_news_rss(query, page_size=page_size)
    merged_articles = dedupe_articles([*primary_articles, *rss_articles])

    if merged_articles:
        return merged_articles[:page_size], None
    return [], primary_error or rss_error or "No evidence sources returned articles."


def fetch_latest_news(query: str | None = None, page_size: int = 9) -> tuple[list[dict[str, str]], str | None]:
    query = (query or NEWS_DEFAULT_QUERY).strip()
    page_size = max(1, min(page_size, 20))
    cache_key = f"{query}:{page_size}"
    now = time.time()
    cached_item = _news_cache.get(cache_key)
    if cached_item is not None and now < float(cached_item["expires_at"]):
        return cached_item["payload"], None

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        articles, rss_error = fetch_google_news_rss(query, page_size)
        return articles, None if articles else f"NEWS_API_KEY is not configured. {rss_error or ''}".strip()

    from_time = (datetime.now(timezone.utc) - timedelta(hours=48)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    try:
        response = requests.get(
            NEWS_API_URL,
            params={
                "q": query,
                "searchIn": "title,description",
                "language": "en",
                "sortBy": "publishedAt",
                "from": from_time,
                "pageSize": page_size,
                "apiKey": api_key,
            },
            timeout=8,
        )
        data = response.json()
        if response.status_code >= 400 or data.get("status") == "error" or response.status_code == 429:
            newsapi_error = data.get("message") or "Latest news could not be loaded right now."
            articles, rss_error = fetch_google_news_rss(query, page_size)
            return articles, None if articles else newsapi_error or rss_error
        articles = data.get("articles", [])
    except requests.RequestException:
        articles, rss_error = fetch_google_news_rss(query, page_size)
        return articles, None if articles else rss_error or "Latest news could not be loaded right now."
    except ValueError:
        articles, rss_error = fetch_google_news_rss(query, page_size)
        return articles, None if articles else rss_error or "NewsAPI returned an invalid response."

    payload = []
    for article in articles:
        title = article.get("title") or "Untitled story"
        if title == "[Removed]":
            continue
        description = article.get("description") or ""
        if not has_enough_article_text(title, description):
            continue
        payload.append(
            {
                "title": title,
                "description": description,
                "content": article.get("content") or description or title,
                "source": (article.get("source") or {}).get("name") or "Unknown source",
                "image": article.get("urlToImage") or DEFAULT_NEWS_IMAGE,
                "publishedAt": article.get("publishedAt") or "",
                "url": article.get("url") or "",
                "provider": "NewsAPI",
            }
        )

    if not payload:
        articles, rss_error = fetch_google_news_rss(query, page_size)
        if articles:
            payload = articles
        elif rss_error:
            return [], rss_error

    _news_cache[cache_key] = {"payload": payload, "expires_at": now + NEWS_CACHE_SECONDS}
    return payload, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    source = (data.get("source") or "").strip()
    url = (data.get("url") or "").strip()

    if len(text) < 20:
        return jsonify({"error": "Please enter at least 20 characters of news text."}), 400

    try:
        return jsonify(build_verdict(text, source=source, url=url))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


@app.route("/news")
def latest_news():
    query = (request.args.get("q") or "").strip()
    try:
        page_size = int(request.args.get("pageSize", "9"))
    except ValueError:
        page_size = 9
    articles, error = fetch_latest_news(query=query, page_size=page_size)
    status = 503 if error and not articles else 200
    return jsonify({"articles": articles, "error": error, "query": query or NEWS_DEFAULT_QUERY}), status


@app.route("/news_page")
def news_page():
    page = request.args.get("page", 1, type=int)
    q = request.args.get("q", "")
    page_size = 20
    articles, error = fetch_latest_news(q, page_size)
    return render_template("news_fixed.html", articles=articles, error=error, query=q, page=page, page_size=page_size)


@app.route("/topic", methods=["POST"])
def topic():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    return jsonify({"topic": extract_topic(text)})


try:
    load_artifacts()
except FileNotFoundError:
    app.logger.warning("Model artifacts are missing. Run `python model.py` before production use.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
