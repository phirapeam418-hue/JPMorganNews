"""Free tech news bot: RSS -> full article -> Gemini (score + dedupe) -> Telegram.

Modes (GitHub Actions runs these for you):
    python bot.py alert    # hourly: read your buttons/replies, fetch news, send important ones
    python bot.py digest   # daily 08:00: send medium-importance stories
    python bot.py quiz     # daily 20:15: vocabulary quiz
    python bot.py weekly   # Sunday: week in review

In Telegram you can:
    press 👍 / 👎 under a story   -> the bot learns what you like
    press 📖 More                 -> longer explanation in simple English
    reply to a story with a question -> the bot answers about that article
    /quiz  /level C1  /status  /help
The bot checks for your messages each time it runs (about once an hour).
"""
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
import trafilatura
from google import genai
from google.genai import types

# Windows consoles default to a code page without Thai; avoid crashes when printing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ======================= Settings you can change =======================
FEEDS = [
    "https://www.datacenterdynamics.com/en/rss/",
    "https://www.theregister.com/headlines.atom",
    "https://blog.cloudflare.com/rss/",
    "https://news.ycombinator.com/rss",   # official Hacker News front page (more reliable than hnrss.org)
    "https://feeds.arstechnica.com/arstechnica/index",
]

# Describe your job and interests. This drives the "impact" score.
MY_FIELD = (
    "I work in data center / infrastructure operations. I care about: outages, "
    "cloud and data center news, cooling and power, networking, security "
    "vulnerabilities affecting servers, DevOps tools, and major AI infrastructure news."
)

# Stories whose title or summary contain any of these words are ALWAYS sent right away.
WATCHLIST = ["outage", "zero-day", "actively exploited", "ransomware"]

# Importance criteria (each scored 1-10 by Gemini). Raise a weight to make it count more.
CRITERIA = {"impact": 1.0, "urgency": 1.0, "scale": 1.0, "credibility": 1.0}

ALERT_MIN = 8             # score >= this: send right away
DIGEST_MIN = 5            # score >= this: daily digest at 08:00
MAX_ALERTS_PER_RUN = 5    # extra urgent stories move to the digest instead of spamming you
DIGEST_MAX = 8            # stories in the daily digest
WEEKLY_TOP = 5            # stories in the Sunday "week in review"

QUIET_HOURS = (23, 7)     # no alerts 23:00-07:00 (Thailand time); held until morning

VOCAB_PER_STORY = 3       # vocabulary words shown under each story
VOCAB_LEVEL = "B2"        # starting CEFR level of words to learn: B1, B2, C1 or C2
AUTO_LEVEL = True         # move the level up/down automatically from your quiz results
LEVEL_UP_AT = 0.85        # >= 85% correct in the last LEVEL_WINDOW answers -> harder words
LEVEL_DOWN_AT = 0.50      # < 50% correct -> easier words
LEVEL_WINDOW = 20
MASTERED_AFTER = 3        # a word is "mastered" after 3 correct answers in a row (no more quizzes)
QUIZ_QUESTIONS = 5        # quiz questions per day (daily quiz + /quiz)
QUIZ_MIN_WORDS = 4        # need at least this many saved words before a quiz can run
LEVELS = ["B1", "B2", "C1", "C2"]

MAX_NEW_PER_RUN = 24      # new articles analysed per run (protects the free quota)
BATCH_SIZE = 8            # articles per Gemini request (fewer requests = less quota)
ARTICLE_CHARS = 3000      # how much of each article Gemini reads
FAIL_ALERT_AFTER = 3      # warn you after this many failed runs in a row

# First choice. If it is busy or gone, the bot automatically tries other free "flash" models
# your key can use. You can list several, comma-separated, in the GEMINI_MODEL variable.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
TZ = timezone(timedelta(hours=7))
STATE_FILE = Path(__file__).parent / "state.json"
# ======================================================================

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
NOTION_DB_ID = os.environ.get("NOTION_DATABASE_ID", "")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# ----------------------------- state -----------------------------
def load_state():
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:  # damaged file: keep a copy and start fresh instead of crashing
            backup = STATE_FILE.with_name("state.broken.json")
            STATE_FILE.replace(backup)
            print(f"state.json was damaged ({e}); moved it to {backup.name} and started fresh")
    defaults = {"started": False, "seen": [], "stories": [], "next_id": 1, "vocab": [],
                "feedback": {"liked": [], "disliked": []}, "tg_offset": 0,
                "polls": {}, "fail": {}, "level": VOCAB_LEVEL, "answers": []}
    for k, v in defaults.items():
        state.setdefault(k, v)
    for v in state["vocab"]:
        for k, d in {"level": "", "example": "", "quizzed": 0, "wrong": 0,
                     "streak": 0, "mastered": False}.items():
            v.setdefault(k, d)
    return state


def save_state(state):
    cutoff = (now() - timedelta(days=14)).isoformat()
    state["stories"] = [s for s in state["stories"] if s["ts"] >= cutoff]
    state["seen"] = state["seen"][-3000:]
    state["vocab"] = state["vocab"][-500:]
    state["polls"] = dict(list(state["polls"].items())[-50:])
    for k in ("liked", "disliked"):
        state["feedback"][k] = state["feedback"][k][-20:]
    tmp = STATE_FILE.with_name("state.tmp")  # write to a temp file first, so a crash can't empty state.json
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_FILE)


def now():
    return datetime.now(TZ)


def is_quiet():
    start, end = QUIET_HOURS
    h = now().hour
    return (h >= start or h < end) if start > end else (start <= h < end)


def find_story(state, sid):
    return next((s for s in state["stories"] if s["id"] == sid), None)


# ----------------------------- health -----------------------------
HEALTH_NOTES = []  # collected during a run, sent as ONE message at the end


def note_fail(state, key, err):
    n = state["fail"].get(key, 0) + 1
    state["fail"][key] = n
    print(f"[fail] {key}: {err}")
    if n == FAIL_ALERT_AFTER:
        HEALTH_NOTES.append(f"⚠️ '{key}' failed {n} runs in a row: {str(err)[:200]}")


def note_ok(state, key):
    if state["fail"].get(key):
        if state["fail"][key] >= FAIL_ALERT_AFTER:
            HEALTH_NOTES.append(f"✅ '{key}' is working again.")
        state["fail"][key] = 0


def flush_health():
    if HEALTH_NOTES:
        try:
            tg("sendMessage", chat_id=TG_CHAT_ID, text="Bot health\n" + "\n".join(HEALTH_NOTES)[:3900])
        except Exception as e:
            print(f"could not send health note: {e}")
        HEALTH_NOTES.clear()


# ----------------------------- Telegram -----------------------------
def tg(method, **params):
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/{method}", json=params, timeout=20)
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method}: {data.get('description')}")
    return data["result"]


def buttons(sid, chosen=None):
    up = "✅ 👍" if chosen == "up" else "👍"
    down = "✅ 👎" if chosen == "down" else "👎"
    return {"inline_keyboard": [[
        {"text": up, "callback_data": f"up:{sid}"},
        {"text": down, "callback_data": f"down:{sid}"},
        {"text": "📖 More", "callback_data": f"more:{sid}"},
    ]]}


def format_story(s):
    icon = "🔴" if s["score"] >= ALERT_MIN else "🟡"
    tag = " 👀 watchlist" if s.get("watch") else ""
    breakdown = " ".join(f"{k[0].upper()}{s.get(k, '?')}" for k in CRITERIA)
    lines = [f"{icon} [{s['score']}/10]{tag} {s['title']}",
             f"({breakdown}) {s.get('reason', '')}", "", s.get("summary", "")]
    if s.get("vocab"):
        lines += [f"• {v['word']}" + (f" ({v['level']})" if v.get("level") else "") + f" = {v['thai']}"
                  for v in s["vocab"]]
    lines.append(s["link"])
    if s.get("extra_links"):
        lines.append("Also reported by:")
        lines += s["extra_links"][:3]
    return "\n".join(lines)[:4000]


def send_story(state, s):
    msg = tg("sendMessage", chat_id=TG_CHAT_ID, text=format_story(s),
             reply_markup=buttons(s["id"]), link_preview_options={"is_disabled": True})
    s["msg_id"] = msg["message_id"]
    s["status"] = "sent"
    s["sent_ts"] = now().isoformat()
    add_vocab(state, s.get("vocab", []))
    notion_save(state, s)


def add_vocab(state, words):
    known = {v["word"].lower() for v in state["vocab"]}
    for w in words:
        if w.get("word") and w.get("thai") and w["word"].lower() not in known:
            state["vocab"].append({"word": w["word"], "thai": w["thai"], "level": w.get("level", ""),
                                   "example": w.get("example", ""), "quizzed": 0, "wrong": 0,
                                   "streak": 0, "mastered": False})
            known.add(w["word"].lower())


# ----------------------------- Notion (optional) -----------------------------
def notion_save(state, s):
    if not (NOTION_TOKEN and NOTION_DB_ID):
        return
    vocab = ", ".join(f"{v['word']} = {v['thai']}" for v in s.get("vocab", []))
    body = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Name": {"title": [{"text": {"content": s["title"][:200]}}]},
            "URL": {"url": s["link"]},
            "Score": {"number": s["score"]},
            "Summary": {"rich_text": [{"text": {"content": s.get("summary", "")[:1900]}}]},
            "Vocab": {"rich_text": [{"text": {"content": vocab[:1900]}}]},
            "Date": {"date": {"start": now().date().isoformat()}},
        },
    }
    try:
        r = requests.post("https://api.notion.com/v1/pages", json=body, timeout=20, headers={
            "Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": "2022-06-28"})
        r.raise_for_status()
        note_ok(state, "notion")
    except Exception as e:
        detail = e.response.text if getattr(e, "response", None) is not None else e
        note_fail(state, "notion", detail)


# ----------------------------- Gemini -----------------------------
_MODELS = []  # models to try, in order (filled on first use)


def candidate_models():
    """Your chosen model(s) first, then other text 'flash' models this key can use."""
    if _MODELS:
        return _MODELS
    names = [m.strip() for m in MODEL.split(",") if m.strip()]
    try:
        found = []
        for m in client.models.list():
            n = m.name.replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            skip = ("image", "tts", "audio", "live", "embed", "vision", "thinking-exp")
            if "flash" in n and "generateContent" in actions and not any(x in n for x in skip):
                found.append(n)
        stable = sorted([n for n in found if "preview" not in n and "exp" not in n], reverse=True)
        preview = sorted([n for n in found if n not in stable], reverse=True)
        names += stable + preview
    except Exception as e:
        print(f"could not list models: {e}")
    seen = set()
    _MODELS.extend(n for n in names if not (n in seen or seen.add(n)))
    del _MODELS[6:]  # don't try forever
    return _MODELS


def gemini(prompt, json_mode=False):
    cfg = {"temperature": 0.2, "automatic_function_calling": {"disable": True}}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    last = None
    for model in list(candidate_models()):
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
                text = resp.text or ""
                result = json.loads(text) if json_mode else text.strip()
                if _MODELS and _MODELS[0] != model:  # use the working model first next time
                    _MODELS.remove(model)
                    _MODELS.insert(0, model)
                    print(f"[gemini] switched to {model}")
                return result
            except Exception as e:
                last = e
                msg = str(e)
                if any(c in msg for c in ("503", "UNAVAILABLE", "500", "INTERNAL")) and attempt == 0:
                    time.sleep(15)  # busy: wait once, then try the next model
                    continue
                if any(c in msg for c in ("404", "NOT_FOUND", "429", "RESOURCE_EXHAUSTED",
                                          "503", "UNAVAILABLE", "500", "INTERNAL", "403",
                                          "PERMISSION_DENIED")) or isinstance(e, json.JSONDecodeError):
                    print(f"[gemini] {model}: {msg[:120]} -> trying next model")
                    break  # next model (each model has its own free quota)
                raise  # e.g. bad API key: no point trying other models
    raise RuntimeError(f"all Gemini models failed. Last error: {last}")


def run_models(state):
    """python bot.py models  -> show which Gemini models work with your key."""
    print("Testing models (this takes a minute)...")
    for m in candidate_models():
        try:
            r = client.models.generate_content(model=m, contents="Reply with the word OK.")
            print(f"  OK    {m}: {(r.text or '').strip()[:20]}")
        except Exception as e:
            print(f"  FAIL  {m}: {str(e)[:100]}")


def total_score(r):
    w = sum(CRITERIA.values())
    return round(sum(int(r.get(k, 1)) * v for k, v in CRITERIA.items()) / w)


BATCH_PROMPT = """You are a news editor for this reader:
{field}
{feedback}
For each NEW article below:
1. Score 1-10: impact (how directly it affects the reader's work), urgency (happening now / act soon),
   scale (how many people or companies affected), credibility (1 = rumor/clickbait, 10 = official/widely confirmed).
2. duplicate_of: if it reports the SAME event as a RECENT story, give that id (e.g. "R12").
   If it reports the same event as an earlier NEW article, give that id (e.g. "N1"). Otherwise null.
3. reason: one short English sentence explaining the scores.
4. summary: 2 simple English sentences (CEFR B1-B2) on what happened and why it matters.
5. vocab: {nvocab} words or phrases from the article for a Thai learner at CEFR {level}.
   Pick words at level {level} or harder. Do NOT pick everyday words (A1-B1, e.g. "company", "system", "use"),
   acronyms, product names, or any of these already-learned words: {avoid}.
   Prefer words useful in professional English beyond this article; at most one technical term.
   Format: {{"word": "...", "thai": "Thai meaning", "level": "B2/C1/C2",
   "example": "one short sentence using the word, from the article if possible"}}.

Return ONLY JSON: {{"results": [{{"id": "N1", "impact": 0, "urgency": 0, "scale": 0, "credibility": 0,
"duplicate_of": null, "reason": "", "summary": "", "vocab": []}}]}}

RECENT stories:
{recent}

NEW articles:
{articles}"""


def feedback_text(state):
    fb = state["feedback"]
    if not (fb["liked"] or fb["disliked"]):
        return ""
    out = "Reader feedback (use it to judge impact):"
    if fb["liked"]:
        out += "\nLiked: " + " | ".join(fb["liked"][-10:])
    if fb["disliked"]:
        out += "\nNot interested: " + " | ".join(fb["disliked"][-10:])
    return out


def analyze_batch(state, batch):
    cutoff = (now() - timedelta(hours=48)).isoformat()
    recent = [s for s in state["stories"] if s["ts"] >= cutoff]
    recent_txt = "\n".join(f"R{s['id']}: {s['title']}" for s in recent[-40:]) or "(none)"
    arts = "\n\n".join(f"N{i+1}: [{a['source']}] {a['title']}\n{a['body']}" for i, a in enumerate(batch))
    avoid = ", ".join(v["word"] for v in state["vocab"][-80:]) or "(none)"
    prompt = BATCH_PROMPT.format(field=MY_FIELD, feedback=feedback_text(state),
                                 nvocab=VOCAB_PER_STORY, level=state["level"], avoid=avoid,
                                 recent=recent_txt, articles=arts)
    data = gemini(prompt, json_mode=True)
    results = data.get("results", data) if isinstance(data, dict) else data
    return {r.get("id"): r for r in results if isinstance(r, dict)}


# ----------------------------- fetching -----------------------------
def clean(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def fetch_new(state):
    seen, items = set(state["seen"]), []
    for url in FEEDS:
        try:
            feed = feedparser.parse(url, agent="Mozilla/5.0 (tech-news-bot)")
            if not feed.entries:
                raise RuntimeError(f"no entries (status {getattr(feed, 'status', '?')})")
            note_ok(state, url)
        except Exception as e:
            note_fail(state, url, e)
            continue
        source = feed.feed.get("title", url)
        for e in feed.entries[:15]:
            link = e.get("link")
            if link and link not in seen:
                items.append({"link": link, "title": e.get("title", ""), "source": source,
                              "rss": clean(e.get("summary", ""))[:1500]})
                seen.add(link)
    return items


def full_text(link, fallback=""):
    try:
        r = requests.get(link, timeout=15, headers={"User-Agent": "Mozilla/5.0 (tech-news-bot)"})
        txt = trafilatura.extract(r.text) or ""
    except Exception:
        txt = ""
    return (txt if len(txt) > len(fallback) else fallback)[:ARTICLE_CHARS]


def on_watchlist(item):
    hay = f"{item['title']} {item['rss']}".lower()
    return any(re.search(rf"\b{re.escape(w.lower())}\b", hay) for w in WATCHLIST)


# ----------------------------- modes -----------------------------
def run_alert(state):
    items = fetch_new(state)

    if not state["started"]:  # first run: remember current news, don't flood you
        tg("sendMessage", chat_id=TG_CHAT_ID,
           text=f"✅ Bot started. Watching {len(FEEDS)} sources. New stories will arrive from the next run.")
        state["seen"] += [i["link"] for i in items]  # only after the message was really sent
        state["started"] = True
        return

    # send stories held during quiet hours
    alerts_sent = 0
    if not is_quiet():
        for s in [s for s in state["stories"] if s["status"] == "night"]:
            if alerts_sent < MAX_ALERTS_PER_RUN:
                send_story(state, s)
                alerts_sent += 1
            else:
                s["status"] = "queued"

    items = items[:MAX_NEW_PER_RUN]
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        for a in batch:
            a["body"] = full_text(a["link"], a["rss"])
        try:
            results = analyze_batch(state, batch)
            note_ok(state, "gemini")
        except Exception as e:
            note_fail(state, "gemini", e)
            break  # unseen items are retried next run

        made = {}  # "N1" -> story created in this batch
        for i, a in enumerate(batch):
            nid = f"N{i+1}"
            state["seen"].append(a["link"])
            r = results.get(nid)
            if not r:
                continue
            dup = (r.get("duplicate_of") or "").strip()
            target = None
            if dup.startswith("R") and dup[1:].isdigit():
                target = find_story(state, int(dup[1:]))
            elif dup in made:
                target = made[dup]
            if target:
                target.setdefault("extra_links", []).append(a["link"])
                continue

            score = total_score(r)
            watch = on_watchlist(a)
            if score < DIGEST_MIN and not watch:
                continue
            s = {"id": state["next_id"], "ts": now().isoformat(), "title": a["title"],
                 "link": a["link"], "source": a["source"], "score": score, "watch": watch,
                 "reason": r.get("reason", ""), "summary": r.get("summary", ""),
                 "vocab": (r.get("vocab") or [])[:VOCAB_PER_STORY], "status": "queued",
                 **{k: r.get(k) for k in CRITERIA}}
            state["next_id"] += 1
            state["stories"].append(s)
            made[nid] = s

            if score >= ALERT_MIN or watch:
                if is_quiet():
                    s["status"] = "night"
                elif alerts_sent < MAX_ALERTS_PER_RUN:
                    send_story(state, s)
                    alerts_sent += 1
        time.sleep(4)


def run_digest(state):
    queued = sorted([s for s in state["stories"] if s["status"] == "queued"],
                    key=lambda s: -s["score"])
    if not queued:
        return
    tg("sendMessage", chat_id=TG_CHAT_ID, text=f"📰 Daily Tech Digest ({len(queued[:DIGEST_MAX])} stories)")
    for s in queued[:DIGEST_MAX]:
        send_story(state, s)
    for s in queued[DIGEST_MAX:]:
        s["status"] = "skipped"


def run_weekly(state):
    cutoff = (now() - timedelta(days=7)).isoformat()
    week = sorted([s for s in state["stories"] if s["ts"] >= cutoff and s["status"] == "sent"],
                  key=lambda s: -s["score"])[:WEEKLY_TOP]
    if week:
        lines = ["🗓 Week in review: top stories"]
        lines += [f"{i+1}. [{s['score']}] {s['title']}\n{s['link']}" for i, s in enumerate(week)]
        mastered = sum(v["mastered"] for v in state["vocab"])
        lines.append(f"📚 Words saved: {len(state['vocab'])} | mastered: {mastered} | "
                     f"level: {state['level']}")
        tg("sendMessage", chat_id=TG_CHAT_ID, text="\n\n".join(lines),
           link_preview_options={"is_disabled": True})


def blank_sentence(v):
    """Example sentence with the word replaced by ____ (None if the word isn't in it)."""
    ex = v.get("example") or ""
    pat = re.compile(rf"\b{re.escape(v['word'])}\b", re.I)
    return pat.sub("_____", ex, count=1) if pat.search(ex) else None


def run_quiz(state):
    pool = [v for v in state["vocab"] if not v["mastered"]]
    everyone = state["vocab"]
    if len(pool) < 1 or len(everyone) < QUIZ_MIN_WORDS:
        tg("sendMessage", chat_id=TG_CHAT_ID,
           text=f"📚 Not enough words for a quiz yet ({len(pool)} to learn, need {QUIZ_MIN_WORDS}).")
        return
    # words you got wrong first, then words never quizzed, then the rest
    order = sorted(pool, key=lambda v: (-v["wrong"], v["quizzed"], random.random()))
    picks = order[:min(QUIZ_QUESTIONS, len(pool))]
    tg("sendMessage", chat_id=TG_CHAT_ID,
       text=f"📚 Daily vocabulary quiz: {len(picks)} questions (level {state['level']})")
    for i, v in enumerate(picks):
        blank = blank_sentence(v)
        if blank and i % 2 == 1:  # every second question: fill in the blank (English options)
            others = list({w["word"] for w in everyone if w["word"].lower() != v["word"].lower()})
            answer, question = v["word"], f"Fill in the blank:\n{blank}"
        else:                     # meaning question (Thai options)
            others = list({w["thai"] for w in everyone if w["thai"] != v["thai"]})
            answer, question = v["thai"], f'What does "{v["word"]}" mean?'
            if v.get("example"):
                question += f'\n"{v["example"]}"'
        opts = random.sample(others, min(3, len(others))) + [answer]
        random.shuffle(opts)
        poll = tg("sendPoll", chat_id=TG_CHAT_ID, question=question[:300],
                  options=[{"text": o[:100]} for o in opts], type="quiz",
                  correct_option_id=opts.index(answer), is_anonymous=False,
                  explanation=f'{v["word"]} = {v["thai"]}'[:200])
        state["polls"][poll["poll"]["id"]] = {"word": v["word"], "correct": opts.index(answer)}
        v["quizzed"] += 1
        time.sleep(1)


def record_answer(state, v, correct):
    if correct:
        v["streak"] += 1
        v["wrong"] = max(0, v["wrong"] - 1)
        if v["streak"] >= MASTERED_AFTER and not v["mastered"]:
            v["mastered"] = True
            tg("sendMessage", chat_id=TG_CHAT_ID, text=f'🏆 Mastered: "{v["word"]}"')
    else:
        v["streak"] = 0
        v["wrong"] += 1

    state["answers"] = (state["answers"] + [bool(correct)])[-LEVEL_WINDOW:]
    if not AUTO_LEVEL or len(state["answers"]) < LEVEL_WINDOW:
        return
    acc = sum(state["answers"]) / len(state["answers"])
    i = LEVELS.index(state["level"]) if state["level"] in LEVELS else 1
    new = None
    if acc >= LEVEL_UP_AT and i < len(LEVELS) - 1:
        new = LEVELS[i + 1]
    elif acc < LEVEL_DOWN_AT and i > 0:
        new = LEVELS[i - 1]
    if new:
        up = LEVELS.index(new) > i
        state["level"], state["answers"] = new, []
        tg("sendMessage", chat_id=TG_CHAT_ID, text=(
            f"{'⬆️' if up else '⬇️'} {round(acc * 100)}% correct in your last {LEVEL_WINDOW} answers. "
            f"New words will now be level {new}."))


# ----------------------------- your buttons / replies -----------------------------
def answer_about(s, question):
    body = full_text(s["link"], s.get("summary", ""))
    prompt = (f"Article title: {s['title']}\nArticle text:\n{body}\n\n"
              f"{question}\nUse simple English (CEFR B1-B2). If the article does not say, say so.")
    return gemini(prompt)


def handle_updates(state):
    try:
        updates = tg("getUpdates", offset=state["tg_offset"], timeout=0,
                     allowed_updates=["message", "callback_query", "poll_answer"])
        note_ok(state, "telegram")
    except Exception as e:
        note_fail(state, "telegram", e)
        return
    for u in updates:
        state["tg_offset"] = u["update_id"] + 1
        try:
            handle_one(state, u)
        except Exception as e:
            print(f"update error: {e}")


def handle_one(state, u):
    if "callback_query" in u:
        cq = u["callback_query"]
        if str(cq["from"]["id"]) != TG_CHAT_ID:
            return
        try:
            tg("answerCallbackQuery", callback_query_id=cq["id"])
        except Exception:
            pass  # too old to answer; still process it
        action, sid = cq["data"].split(":")
        s = find_story(state, int(sid))
        if not s:
            return
        if action in ("up", "down"):
            key = "liked" if action == "up" else "disliked"
            state["feedback"][key].append(s["title"])
            tg("editMessageReplyMarkup", chat_id=TG_CHAT_ID, message_id=s["msg_id"],
               reply_markup=buttons(s["id"], action))
        elif action == "more":
            text = answer_about(s, "Explain this story in 5-6 sentences: what happened, why, "
                                   "and what it means for someone working in: " + MY_FIELD)
            tg("sendMessage", chat_id=TG_CHAT_ID, text=f"📖 {text}"[:4000],
               reply_parameters={"message_id": s["msg_id"]})

    elif "poll_answer" in u:
        pa = u["poll_answer"]
        q = state["polls"].pop(pa["poll_id"], None)
        if not q or str(pa["user"]["id"]) != TG_CHAT_ID:
            return
        v = next((v for v in state["vocab"] if v["word"] == q["word"]), None)
        if v:
            record_answer(state, v, bool(pa["option_ids"]) and pa["option_ids"][0] == q["correct"])

    elif "message" in u:
        m = u["message"]
        if str(m["chat"]["id"]) != TG_CHAT_ID:
            return  # ignore strangers
        text = (m.get("text") or "").strip()
        reply_to = m.get("reply_to_message", {}).get("message_id")
        story = reply_to and next((s for s in state["stories"] if s.get("msg_id") == reply_to), None)
        if story and text:
            ans = answer_about(story, f"Answer the reader's question about this article: {text}")
            tg("sendMessage", chat_id=TG_CHAT_ID, text=ans[:4000],
               reply_parameters={"message_id": m["message_id"]})
        elif text.startswith("/quiz"):
            run_quiz(state)
        elif text.startswith("/level"):
            arg = text[6:].strip().upper()
            if arg in LEVELS:
                state["level"], state["answers"] = arg, []
                tg("sendMessage", chat_id=TG_CHAT_ID, text=f"OK, new words will be level {arg}.")
            else:
                tg("sendMessage", chat_id=TG_CHAT_ID,
                   text=f"Current level: {state['level']}. Use /level B1, /level B2, /level C1 or /level C2")
        elif text.startswith("/status"):
            queued = sum(s["status"] == "queued" for s in state["stories"])
            problems = [k for k, n in state["fail"].items() if n]
            ans = state["answers"]
            acc = f"{round(100 * sum(ans) / len(ans))}% of last {len(ans)}" if ans else "no answers yet"
            tg("sendMessage", chat_id=TG_CHAT_ID, text=(
                f"Stories waiting for digest: {queued}\n"
                f"Words: {len(state['vocab'])} saved, {sum(v['mastered'] for v in state['vocab'])} mastered\n"
                f"Level: {state['level']} | quiz accuracy: {acc}\n"
                f"Problems: {', '.join(problems) or 'none'}"))
        else:
            tg("sendMessage", chat_id=TG_CHAT_ID, text=(
                "I check messages about once an hour.\n"
                "• Reply to a story with a question\n• 👍/👎 to teach me what you like\n"
                "• /quiz  vocabulary quiz now\n• /level C1  change word level\n• /status  bot status"))


# ----------------------------- main -----------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "alert"
    state = load_state()
    try:
        if mode != "models":
            handle_updates(state)
        {"alert": run_alert, "digest": run_digest, "weekly": run_weekly, "quiz": run_quiz,
         "models": run_models, "updates": lambda s: None}[mode](state)
    finally:
        flush_health()
        save_state(state)
