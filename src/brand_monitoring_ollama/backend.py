from __future__ import annotations
import html
import json
import logging
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable, Literal, get_args
import ollama
import pandas as pd
import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
logger = logging.getLogger(__name__)
# ---------------------------------settings------------------------------------------ #

DB_PATH = os.getenv("BRAND_MONITOR_DB", "brand_monitor.db")
# .strip(): the old constant had a trailing space inside the model name.
MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud").strip()
HN_ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"
HN_PAGE_SIZE = 50
HN_MAX_PAGES = 20            # Algolia serves at most 1000 hits per query
MAX_ANALYSIS_CHARS = 1500    # long comments are cut before classification
SUMMARY_CHAR_BUDGET = 6000   # total comment text sent to a summary prompt
SUMMARY_ITEM_CHARS = 500     # per-comment cap inside a summary prompt
_report_error: Callable[[str], object] = logger.error

def set_error_handler(handler: Callable[[str], object]) -> None:
    """Send backend errors to the UI, e.g. set_error_handler(st.error)."""
    global _report_error
    _report_error = handler
# --------------------------------Database------------------------------------------- #

@contextmanager

def _db():
    """Open a connection, commit on success, roll back on error, always close."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db() -> None:
    """Create the table and the indexes that make duplicate checks free."""
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                url TEXT,
                timestamp DATETIME,
                sentiment TEXT,
                topic TEXT,
                urgency TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mentions_brand_time ON mentions(brand, timestamp)"
        )
        already_unique = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'idx_mentions_brand_url'"
        ).fetchone()
        if not already_unique:
            # Databases made by the old version can hold duplicates, which would
            # make the unique index fail. Keep the oldest copy of each.
            conn.execute(
                """
                DELETE FROM mentions
                WHERE url IS NOT NULL
                  AND id NOT IN (
                      SELECT MIN(id) FROM mentions WHERE url IS NOT NULL GROUP BY brand, url
                  )
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX idx_mentions_brand_url ON mentions(brand, url)"
            )

def _db_time(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)

def _insert_many(rows: list[tuple]) -> int:
    """Insert rows in one transaction; duplicates are skipped by the unique index."""
    if not rows:
        return 0
    with _db() as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO mentions (brand, source, text, url, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes - before

def add_mention(brand_name, source, text, url, timestamp) -> bool:
    """Store one mention. Returns True if it was new, False if it already existed."""
    return _insert_many([(brand_name, source, text, url, _db_time(timestamp))]) == 1

def _parse_times(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="ISO8601")
    except (TypeError, ValueError):  # older pandas without format="ISO8601"
        return pd.to_datetime(series, errors="coerce")

def get_all_mentions_as_df(brand_name: str) -> pd.DataFrame:
    """Every stored mention for one brand, newest first."""
    with _db() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM mentions WHERE brand = ? ORDER BY timestamp DESC",
            conn,
            params=(brand_name,),
        )
    df["timestamp"] = _parse_times(df["timestamp"])
    return df

def update_mention_analysis(mention_id, sentiment, topic, urgency) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE mentions SET sentiment = ?, topic = ?, urgency = ? WHERE id = ?",
            (sentiment, topic, urgency, mention_id),
        )
# ---------------------------------Text cleaning------------------------------------------ #

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

def _clean_text(raw) -> str:
    """Strip tags first, then decode entities, then collapse whitespace."""
    if not isinstance(raw, str):
        return ""
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", raw))).strip()

def clean_html(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the frame with the text column cleaned."""
    df = df.copy()
    df["text"] = df["text"].map(_clean_text)
    return df
# ---------------------------------Hacker News collection------------------------------------------ #

@lru_cache(maxsize=1)

def _http() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "brand-monitor/1.0"
    return session

def _hit_to_row(brand: str, hit: dict) -> tuple | None:
    text = _clean_text(hit.get("comment_text"))
    object_id = hit.get("objectID")
    if not text or not object_id:  # deleted or empty comments
        return None
    try:
        created = datetime.fromtimestamp(int(hit["created_at_i"]), tz=timezone.utc)
    except (KeyError, TypeError, ValueError):
        created = datetime.now(timezone.utc)
    return (
        brand,
        "Hacker News",
        text,
        f"https://news.ycombinator.com/item?id={object_id}",
        _db_time(created),
    )

def fetch_hn_mentions(brand_name: str, max_pages: int = 3) -> int:
    """Pull recent Hacker News comments for a brand; returns how many were new."""
    brand_name = brand_name.strip()
    if not brand_name:
        return 0
    pages_to_read = max(1, min(max_pages, HN_MAX_PAGES))
    inserted = 0
    page = 0
    try:
        while page < pages_to_read:
            response = _http().get(
                HN_ENDPOINT,
                params={
                    "query": brand_name,
                    "tags": "comment",
                    "page": page,
                    "hitsPerPage": HN_PAGE_SIZE,
                },
                timeout=(5, 20),
            )
            response.raise_for_status()
            payload = response.json()
            hits = payload.get("hits") or []
            if not hits:
                break
            # Never ask for pages the API says do not exist.
            pages_to_read = min(pages_to_read, payload.get("nbPages") or 0)
            rows = [row for row in (_hit_to_row(brand_name, h) for h in hits) if row]
            inserted += _insert_many(rows)
            page += 1
    except (requests.RequestException, ValueError) as exc:
        _report_error(f"Hacker News request failed: {exc}")
    return inserted

# --------LLM analysis: ONE call per mention returns sentiment, topic and urgency-------#
Sentiment = Literal["Positive", "Negative", "Neutral"]
Topic = Literal[
    "Customer Service Issue",
    "Product Defect/Bug",
    "High Price Complaint",
    "Positive Review",
    "Competitor Comparison",
    "Feature Request",
    "PR/News",
    "Other",
]
Urgency = Literal["High Urgency", "Low Urgency"]

class MentionAnalysis(BaseModel):
    sentiment: Sentiment
    topic: Topic
    urgency: Urgency
ANALYSIS_PROMPT = """
<role>
You are a brand monitoring analyst.
Classify customer comments using the specified labels.
</role>
<task>
Analyze the text and return sentiment, topic, and urgency.
</task>
<classification_rules>
### Sentiment
Choose exactly one:
- Positive
- Negative
- Neutral
### Topic
Choose exactly one:
- Customer Service Issue
- Product Defect/Bug
- High Price Complaint
- Positive Review
- Competitor Comparison
- Feature Request
- PR/News
- Other
### Urgency
Choose exactly one:
- High Urgency
- Low Urgency
Use High Urgency only for serious safety, security,
fraud, widespread failures, or major reputational risks.
Otherwise, choose Low Urgency.
</classification_rules>
<output_format>
Return a valid JSON object containing exactly these keys:
{
    "sentiment": "Positive",
    "topic": "Positive Review",
    "urgency": "Low Urgency"
}
Do not include explanations or Markdown formatting.
</output_format>
"""
_analysis_cache: dict[str, MentionAnalysis] = {}

def _canonical(value, literal_type) -> str | None:
    """Map a loosely written label ('positive ') onto the exact allowed spelling."""
    allowed = {label.lower(): label for label in get_args(literal_type)}
    return allowed.get(str(value).strip().lower())

def _parse_analysis(raw: str) -> MentionAnalysis:
    """Validate a model response and report malformed fields without guessing."""
    try:
        return MentionAnalysis.model_validate_json(raw)
    except ValueError:
        pass
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"Ollama did not return a JSON object: {raw[:500]!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got: {type(data).__name__}")
    normalized = {
        "sentiment": _canonical(data.get("sentiment"), Sentiment),
        "topic": _canonical(data.get("topic"), Topic),
        "urgency": _canonical(data.get("urgency"), Urgency),
    }
    invalid = [key for key, value in normalized.items() if value is None]
    if invalid:
        raise ValueError(
            f"Invalid or missing fields {invalid}; raw Ollama output: {raw[:1000]!r}"
        )
    return MentionAnalysis(**normalized)

def _call_model(text: str) -> MentionAnalysis:
    """Classify one mention; retry once if model violates the response schema."""
    snippet = text[:MAX_ANALYSIS_CHARS]
    cached = _analysis_cache.get(snippet)
    if cached is not None:
        return cached
    messages = [
        {"role": "system", "content": ANALYSIS_PROMPT},
        {"role": "user", "content": snippet},
    ]
    last_error = None
    for attempt in range(2):
        reply = ollama.chat(
            model=MODEL,
            messages=messages,
            format=MentionAnalysis.model_json_schema(),
            options={"temperature": 0},
        )
        raw = reply["message"]["content"]
        try:
            result = _parse_analysis(raw)
        except (ValueError, TypeError) as exc:
            last_error = exc
            if attempt == 0:
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "Your previous JSON did not match the required schema. "
                        "Return exactly one valid JSON object with non-null sentiment, "
                        "topic, and urgency. Use only the enum labels in the schema."
                    )},
                ])
                continue
            break
        _analysis_cache[snippet] = result
        return result
    raise ValueError(f"Ollama returned invalid classification after retry: {last_error}")

def analyze_text(text: str) -> MentionAnalysis | None:
    """Sentiment, topic and urgency for one text, or None if the model failed."""
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return _call_model(text)
    except Exception as exc:
        _report_error(f"Ollama analysis failed: {exc}")
        return None
# The three old entry points stay. They share the cache above, so a UI that
# calls all three on the same text costs one model call instead of three.

def get_sentiment(text):
    result = analyze_text(text)
    return result.sentiment if result else None

def get_topic(text):
    result = analyze_text(text)
    return result.topic if result else None

def get_urgency(text):
    result = analyze_text(text)
    return result.urgency if result else None

def analyze_pending(
    brand_name: str,
    max_workers: int = 4,
    limit: int | None = None,
    on_progress: Callable[[int, int], object] | None = None,
) -> int:
    """Label every unanalysed mention for a brand, several at a time.
    Writes all results in one transaction at the end and returns how many rows
    were updated. Failures are reported once, not once per row.
    """
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, text FROM mentions WHERE brand = ? AND sentiment IS NULL LIMIT ?",
            (brand_name, limit if limit is not None else -1),
        ).fetchall()
    if not rows:
        return 0
    def work(row):
        try:
            return row[0], _call_model(row[1])
        except Exception as exc:
            return row[0], exc
    updates, failures, first_error = [], 0, None
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, row) for row in rows]
        for done, future in enumerate(as_completed(futures), start=1):
            mention_id, outcome = future.result()
            if isinstance(outcome, MentionAnalysis):
                updates.append((outcome.sentiment, outcome.topic, outcome.urgency, mention_id))
            else:
                failures += 1
                first_error = first_error or outcome
            if on_progress is not None:
                on_progress(done, len(rows))
    if updates:
        with _db() as conn:
            conn.executemany(
                "UPDATE mentions SET sentiment = ?, topic = ?, urgency = ? WHERE id = ?",
                updates,
            )
    if failures:
        _report_error(f"{failures} of {len(rows)} mentions could not be analysed: {first_error}")
    return len(updates)
# -------------------------------Summaries-------------------------------------------- #

def _comment_context(texts: list[str]) -> str:
    """Whole comments up to a character budget (the old code cut text mid-sentence)."""
    parts, used = [], 0
    for text in texts:
        piece = text[:SUMMARY_ITEM_CHARS]
        if used + len(piece) > SUMMARY_CHAR_BUDGET:
            break
        parts.append(piece)
        used += len(piece)
    return "\n---\n".join(parts)

def _summarize(
    df: pd.DataFrame,
    mask: pd.Series,
    *,
    role: str,
    guidance: str,
    sections: tuple[str, ...],
    empty_message: str,
    label: str,
) -> str:
    texts = df.loc[mask, "text"].dropna().tolist()  # frame is newest-first
    if not texts:
        return empty_message
    bullets = "\n".join(f"- **{name}:** ..." for name in sections)
    prompt = (
        f"You are {role}.\n{guidance}\n"
        "Use only the comments below and do not invent opinions. Merge similar points, "
        "keep every bullet short and actionable, and say so if the evidence is thin.\n\n"
        f"<comments>\n{_comment_context(texts)}\n</comments>\n\n"
        f"Reply with exactly {len(sections)} Markdown bullets and nothing else:\n{bullets}"
    )
    try:
        reply = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        return reply["message"]["content"].strip()
    except Exception as exc:
        _report_error(f"Ollama error ({label}): {exc}")
        return "Error generating summary."

def generate_positive_report_summary(df: pd.DataFrame) -> str:
    return _summarize(
        df,
        df["sentiment"] == "Positive",
        role="a customer insights specialist studying positive brand feedback",
        guidance=(
            "Find what customers value, what that says about the brand's strengths, "
            "and practical ways to build on it."
        ),
        sections=("Customer Appreciation", "Key Brand Strengths", "Growth Opportunities"),
        empty_message="No positive feedback found to summarize.",
        label="Positive Summary",
    )

def generate_negative_report_summary(df: pd.DataFrame) -> str:
    return _summarize(
        df,
        df["sentiment"] == "Negative",
        role="a customer experience analyst studying complaints",
        guidance=(
            "Find the most frequent problems, any recurring causes or trends, "
            "and the areas that need attention first."
        ),
        sections=("Common Complaints", "Underlying Patterns", "Improvement Areas"),
        empty_message="No negative feedback found to summarize.",
        label="Negative Summary",
    )

def generate_report_summary(df: pd.DataFrame) -> str:
    """Summary of customer suggestions.
    The old version read negative comments here even though its prompt asked
    about suggestions. It now reads comments labelled Feature Request.
    """
    return _summarize(
        df,
        df["topic"] == "Feature Request",
        role="a product strategy analyst studying customer suggestions",
        guidance="Find the recurring feature requests and the product improvements they point to.",
        sections=("Top Suggestions", "Future Opportunities"),
        empty_message="No feature requests found to summarize.",
        label="Suggestions Summary",
    )
