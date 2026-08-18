"""
Prompt Engineering Benchmark Dashboard
======================================
Compare 4 prompting strategies × 3 benchmark models.
LLM-as-judge scoring via a separate Groq-hosted evaluator.

Run:
    streamlit run app.py
"""

import json
import re
import time
from datetime import datetime

import requests
import streamlit as st


class APIError(RuntimeError):
    """Safe, user-facing API error."""


def get_secret(key_name: str) -> str:
    """Safely retrieve a secret from Streamlit Secrets."""
    try:
        return str(st.secrets.get(key_name, "")).strip()
    except Exception:
        return ""


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
MISTRAL_API_KEY = get_secret("MISTRAL_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_JUDGE_API_KEY = get_secret("GROQ_JUDGE_API_KEY")


# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Benchmark Dashboard",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Minimal CSS — only for custom HTML blocks
# the user explicitly requested CSS-styled
# hero, output boxes, score pills, and tables.
# ─────────────────────────────────────────────
st.html("""<style>
  .hero {
    background: linear-gradient(135deg, #0f0e17 0%, #1a1040 60%, #2d1b69 100%);
    color: #fff;
    border-radius: 14px;
    padding: 2rem 2.2rem 1.6rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(124, 106, 247, 0.15);
  }
  .hero-eye {
    font-size: 11px; letter-spacing: .18em; color: #7c6af7;
    text-transform: uppercase; margin-bottom: .5rem; font-weight: 600;
  }
  .hero h1 {
    font-size: 1.85rem; font-weight: 800; letter-spacing: -.02em;
    margin: 0 0 .35rem; line-height: 1.15;
  }
  .hero h1 span { color: #a78bfa; }
  .hero p {
    font-size: 13.5px; color: #9ca3af; margin: 0; line-height: 1.5;
  }

  .output-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: .7rem .85rem;
    font-size: 12.5px;
    line-height: 1.6;
    min-height: 90px;
    max-height: 240px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    color: #d4d4e8;
  }

  .score-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
  }

  .bench-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .bench-table th {
    text-align: center; padding: 8px 10px; color: #888;
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; border-bottom: 2px solid rgba(255,255,255,0.08);
  }
  .bench-table th:first-child { text-align: left; }
  .bench-table td {
    padding: 8px 10px; text-align: center;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }
  .bench-table td:first-child { text-align: left; }
  .bench-table tr:last-child td { border-bottom: none; }
</style>""")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DIMENSIONS = ["Correctness", "Helpfulness", "Clarity", "Conciseness", "Format"]

STRATEGY_META = {
    "zero_shot": {"label": "Zero-shot", "color": "#7c3aed", "icon": ":material/target:"},
    "few_shot": {"label": "Few-shot", "color": "#059669", "icon": ":material/menu_book:"},
    "cot": {"label": "Chain-of-thought", "color": "#d97706", "icon": ":material/psychology:"},
    "structured": {"label": "Structured output", "color": "#2563eb", "icon": ":material/data_object:"},
}

MODEL_META = {
    "gemini": {
        "label": "Gemini 3.6 Flash",
        "provider": "Google",
        "color": "#4285F4",
        "icon": ":material/circle:",
        "api_key": GEMINI_API_KEY,
    },
    "mistral": {
        "label": "Mistral Small",
        "provider": "Mistral AI",
        "color": "#FF7000",
        "icon": ":material/circle:",
        "api_key": MISTRAL_API_KEY,
    },
    "groq_llama": {
        "label": "LLaMA 3.1 8B (Groq)",
        "provider": "Groq",
        "color": "#00C896",
        "icon": ":material/circle:",
        "api_key": GROQ_API_KEY,
    },
}

SCORE_COLOR = {5: "#059669", 4: "#16a34a", 3: "#d97706", 2: "#dc2626", 1: "#991b1b"}

EXAMPLE_TASKS = [
    "Summarize the key differences between RAG and fine-tuning in 3 bullet points.",
    "Explain what a transformer's attention mechanism does to someone who knows Python but not ML.",
    "What are 5 common mistakes junior developers make when building LLM applications?",
    "Write a one-paragraph pitch for a RAG-based legal document assistant for Pakistani law firms.",
    "Given a pandas DataFrame with columns [user_id, session_id, query, response], how would you detect hallucinations at scale?",
]

MAX_COMBINATIONS_PER_RUN = 6
MAX_RUNS_PER_SESSION = 3
MAX_TASK_LENGTH = 2000

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_JUDGE_MODEL = "llama-3.3-70b-versatile"
GROQ_BENCHMARK_MODEL = "llama-3.1-8b-instant"


if "benchmark_runs" not in st.session_state:
    st.session_state.benchmark_runs = 0


def _has_key(value: str) -> bool:
    normalized = (value or "").strip()
    if not normalized:
        return False
    if normalized.startswith("YOUR_"):
        return False
    if normalized.lower() in {"replace_me", "changeme", "change_me"}:
        return False
    if "..." in normalized:
        return False
    return True


def _available_models() -> dict:
    return {k: v for k, v in MODEL_META.items() if _has_key(v["api_key"])}


AVAILABLE_MODELS = _available_models()
JUDGE_AVAILABLE = _has_key(GROQ_JUDGE_API_KEY) or _has_key(GROQ_API_KEY)


def _pretty_error(message: str) -> str:
    return message.strip() or "API error — check your key or rate limits."


def _configured_keys_message() -> str:
    configured = []
    if _has_key(GEMINI_API_KEY):
        configured.append("Gemini")
    if _has_key(MISTRAL_API_KEY):
        configured.append("Mistral")
    if _has_key(GROQ_API_KEY):
        configured.append("Groq benchmark")
    if _has_key(GROQ_JUDGE_API_KEY):
        configured.append("Groq judge")
    return ", ".join(configured) if configured else "none"


# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────
def build_prompt(strategy_key: str, task: str) -> tuple:
    """Returns (system_prompt, user_prompt)."""
    if strategy_key == "zero_shot":
        return (
            "You are a helpful, concise AI assistant. Complete the task exactly as asked.",
            task,
        )

    elif strategy_key == "few_shot":
        return (
            "You are a helpful AI assistant. Study the examples below carefully, then complete the task in the same style.",
            f"""Here are two examples of well-completed tasks:

Example 1
Task: List 3 benefits of using Docker for Python apps.
Output:
1. Consistent environment across dev, staging, and production.
2. Easy dependency isolation — no more "it works on my machine".
3. One-command deployment with docker compose up.

Example 2
Task: Explain gradient descent in one sentence.
Output: Gradient descent iteratively adjusts model weights in the direction that most reduces the loss function, using the gradient as a compass.

Now complete the following task in the same clear, structured style:
Task: {task}
Output:""",
        )

    elif strategy_key == "cot":
        return (
            "You are a careful, analytical AI assistant. Always think step by step before giving your final answer. Show your reasoning process.",
            f"""Think through this step by step, then state your final answer clearly.

Task: {task}

Let's think step by step:""",
        )

    elif strategy_key == "structured":
        return (
            """You are a precise AI assistant. You MUST respond with valid JSON only — no markdown code fences, no preamble, no explanation outside the JSON object.

Use exactly this schema:
{
  "answer": "your main answer here",
  "confidence": 0.0,
  "key_points": ["point 1", "point 2", "point 3"],
  "caveats": "any important limitations or assumptions"
}""",
            f"Complete this task and return JSON only:\n\n{task}",
        )

    return ("You are a helpful assistant.", task)


# ─────────────────────────────────────────────
# API callers
# ─────────────────────────────────────────────
def _requests_error_message(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return "API error — request timed out."
    if isinstance(exc, requests.HTTPError):
        status = getattr(exc.response, "status_code", None)
        if status == 401:
            return "API error — invalid API key."
        if status == 403:
            return "API error — access forbidden."
        if status == 429:
            return "API error — rate limit reached."
        if status in {500, 502, 503}:
            return "API error — Groq service is temporarily unavailable."
        return "API error — check your key or rate limits."
    if isinstance(exc, requests.RequestException):
        return "API error — network or service error."
    return "API error — check your key or rate limits."


def _raise_for_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise APIError(_requests_error_message(exc)) from exc


def _extract_openai_text(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise APIError("API error — malformed response from model.") from exc


def call_gemini(system: str, user: str, api_key: str) -> str:
    """Call Gemini Flash via REST API."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": 1024},
    }
    try:
        response = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
        _raise_for_status(response)
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except KeyError as exc:
        raise APIError("API error — malformed response from Gemini.") from exc
    except requests.RequestException as exc:
        raise APIError(_requests_error_message(exc)) from exc


def call_mistral(system: str, user: str, api_key: str) -> str:
    """Call Mistral Small via Mistral API."""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        _raise_for_status(response)
        return _extract_openai_text(response.json())
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError("API error — malformed response from Mistral.") from exc
    except requests.RequestException as exc:
        raise APIError(_requests_error_message(exc)) from exc


def call_groq(system: str, user: str, api_key: str) -> str:
    """Call LLaMA 3.1 8B via Groq (free tier)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_BENCHMARK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        _raise_for_status(response)
        return _extract_openai_text(response.json())
    except (KeyError, IndexError, TypeError) as exc:
        raise APIError("API error — malformed response from Groq.") from exc
    except requests.RequestException as exc:
        raise APIError(_requests_error_message(exc)) from exc


MODEL_CALLERS = {
    "gemini":     call_gemini,
    "mistral":    call_mistral,
    "groq_llama": call_groq,
}


# ─────────────────────────────────────────────
# LLM-as-judge (uses Groq)
# ─────────────────────────────────────────────
JUDGE_SYSTEM = """You are an objective LLM output evaluator.
Judge only the supplied task and model output.
Do not reward a particular model, provider, verbosity, or hidden reasoning.
Do not reveal or reproduce chain-of-thought.
Evaluate these dimensions independently, each from 1 to 5:
- Correctness
- Helpfulness
- Clarity
- Conciseness
- Format

Return valid JSON only.
No markdown.
No code fences.
No commentary outside the JSON object.
Return exactly these fields:
{
  "Correctness": 1,
  "Helpfulness": 1,
  "Clarity": 1,
  "Conciseness": 1,
  "Format": 1,
  "overall": 1,
  "verdict": "One concise sentence explaining the main strength or weakness."
}"""

JUDGE_JSON_SCHEMA = {
    "name": "judge_output",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Correctness": {"type": "integer", "minimum": 1, "maximum": 5},
            "Helpfulness": {"type": "integer", "minimum": 1, "maximum": 5},
            "Clarity": {"type": "integer", "minimum": 1, "maximum": 5},
            "Conciseness": {"type": "integer", "minimum": 1, "maximum": 5},
            "Format": {"type": "integer", "minimum": 1, "maximum": 5},
            "overall": {"type": "number"},
            "verdict": {"type": "string"},
        },
        "required": [
            "Correctness",
            "Helpfulness",
            "Clarity",
            "Conciseness",
            "Format",
            "overall",
            "verdict",
        ],
    },
    "strict": True,
}


def _strip_code_fences(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _coerce_judge_scores(data: dict) -> dict:
    scores = {}
    for dim in DIMENSIONS:
        value = data.get(dim, 3)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 3.0
        scores[dim] = min(5, max(1, int(round(numeric))))

    overall = data.get("overall")
    try:
        overall_value = float(overall)
    except (TypeError, ValueError):
        overall_value = sum(scores[d] for d in DIMENSIONS) / len(DIMENSIONS)

    verdict = str(data.get("verdict", "Judge returned no verdict.")).strip() or "Judge returned no verdict."
    scores["overall"] = round(overall_value, 2)
    scores["verdict"] = verdict
    return scores


def call_groq_judge(system: str, user: str, api_key: str) -> str:
    """Call Groq judge model via Groq chat completions."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 600, 
        "response_format": {
            "type": "json_object"
        },
    }
    try:
        response = requests.post(GROQ_CHAT_COMPLETIONS_URL, json=payload, headers=headers, timeout=60)
        _raise_for_status(response)
        return _extract_openai_text(response.json())
    except requests.RequestException as exc:
        raise APIError(_requests_error_message(exc)) from exc


def judge_output(task: str, strategy_label: str, output: str) -> dict:
    """Score an output using Groq as the LLM judge."""
    judge_prompt = (
        f"Task given to the model:\n{task}\n\n"
        f"Prompting strategy used: {strategy_label}\n\n"
        f"Model output to evaluate:\n{output}\n\n"
        "Return only the JSON object requested in the system prompt."
    )
    try:
        judge_key = GROQ_JUDGE_API_KEY or GROQ_API_KEY
        raw = call_groq_judge(JUDGE_SYSTEM, judge_prompt, judge_key)
        clean = _strip_code_fences(raw)
        parsed = json.loads(clean)
        if not isinstance(parsed, dict):
            raise ValueError("Judge output is not a JSON object.")
        return _coerce_judge_scores(parsed)
    except (json.JSONDecodeError, ValueError, TypeError, APIError):
        return {
            **{dim: 3 for dim in DIMENSIONS},
            "overall": 3,
            "verdict": "Judge scoring failed; using fallback scores.",
        }


# ─────────────────────────────────────────────
# Token estimator
# ─────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.35))


# ─────────────────────────────────────────────
# Quality chart (grouped bar chart via Altair)
# ─────────────────────────────────────────────
def render_quality_chart(all_scores: dict):
    """Render a grouped bar chart of quality dimensions."""
    import altair as alt
    import pandas as pd

    rows = []
    for combo_key, info in all_scores.items():
        scores = info["scores"]
        m_label = MODEL_META[info["model_key"]]["label"]
        s_label = STRATEGY_META[info["strategy_key"]]["label"]
        label = f"{m_label} / {s_label}"
        for dim in DIMENSIONS:
            rows.append({"Combination": label, "Dimension": dim, "Score": scores.get(dim, 0)})

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Dimension:N", axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5]), title="Score (1–5)"),
            color=alt.Color("Combination:N", scale=alt.Scale(scheme="tableau10")),
            xOffset="Combination:N",
            tooltip=["Combination", "Dimension", "Score"],
        )
        .properties(height=340)
    )
    st.altair_chart(chart)


# ─────────────────────────────────────────────
# Sidebar — strategies & models
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration", anchor=False)

    # ── Strategies ─────────────────────────
    st.subheader("Strategies", anchor=False)
    selected_strategies = []
    for key, meta in STRATEGY_META.items():
        if st.checkbox(
            meta["label"],
            value=(key in ["zero_shot", "cot"]),
            key=f"strat_{key}",
            help=f"Use the {meta['label']} prompting strategy",
        ):
            selected_strategies.append(key)

    # ── Models ─────────────────────────────
    st.subheader("Models", anchor=False)
    if not AVAILABLE_MODELS:
        st.warning(
            "No valid API keys configured. Add real values in Streamlit Secrets.",
            icon=":material/warning:",
        )
    selected_models = []
    for key, meta in MODEL_META.items():
        has_key = key in AVAILABLE_MODELS
        checked = st.checkbox(
            meta["label"],
            value=(key == "gemini" and has_key),
            key=f"model_{key}",
            disabled=not has_key,
            help=f"{meta['provider']}" if has_key else "API key not configured",
        )
        if checked and has_key:
            selected_models.append(key)

    # ── Info ────────────────────────────────
    st.caption(
        "**Rate limits (free tier):**\n"
        "Gemini · 15 req/min\n"
        "Mistral · 1 req/min\n"
        "Groq benchmark · 30 req/min\n"
        "Groq judge · separate key"
    )
    st.caption("Judge scoring uses a separate Groq-hosted evaluator.")
    st.caption(f"Configured secrets: {_configured_keys_message()}")

# Public demo usage
    runs_used = st.session_state.benchmark_runs
    runs_left = max(0, MAX_RUNS_PER_SESSION - runs_used)

    st.divider()

    st.metric(
        "Demo runs remaining",
        f"{runs_left}/{MAX_RUNS_PER_SESSION}",
    )

    st.caption(
        f"Maximum {MAX_COMBINATIONS_PER_RUN} model × strategy "
        "combinations per run."
    )
    st.caption(
        f"LLM judge: {GROQ_JUDGE_MODEL}"
    )


# ─────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eye">Week 5 · Prompt Engineering</div>
  <h1>Prompt strategy <span>benchmark</span></h1>
  <p>Compare prompting strategies × models side-by-side.<br>
  Auto-scored by an LLM judge on 5 quality dimensions.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Task input
# ─────────────────────────────────────────────
task = st.text_area(
    "Task / prompt",
    placeholder='e.g. "Explain what a context window is to a developer who has never used an LLM."',
    height=100,
    max_chars=MAX_TASK_LENGTH,
    label_visibility="collapsed",
    help=f"Maximum {MAX_TASK_LENGTH:,} characters.",
)

# Example tasks
st.caption("**Try an example:**")
example_labels = [f"Example {i+1}" for i in range(len(EXAMPLE_TASKS))]
selected_example = st.pills(
    "Example tasks",
    options=example_labels,
    label_visibility="collapsed",
    help="Click to load an example prompt",
)

if selected_example is not None:
    idx = example_labels.index(selected_example)
    task = EXAMPLE_TASKS[idx]

# inject example from session state (backward compat)
if "_example" in st.session_state:
    task = st.session_state.pop("_example")


# ─────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────
total_combos = len(selected_models) * len(selected_strategies)

too_many_combinations = total_combos > MAX_COMBINATIONS_PER_RUN
runs_exhausted = st.session_state.benchmark_runs >= MAX_RUNS_PER_SESSION
ready = bool(
    task.strip()
    and len(task.strip()) <= MAX_TASK_LENGTH
    and JUDGE_AVAILABLE
    and selected_models
    and selected_strategies
    and not too_many_combinations
    and not runs_exhausted
)

if too_many_combinations:
    st.warning(
        f"You selected {total_combos} combinations. "
        f"For the public demo, please select a maximum of "
        f"{MAX_COMBINATIONS_PER_RUN} combinations per run.",
        icon=":material/warning:",
    )

if runs_exhausted:
    st.error(
        f"You have reached the public demo limit of "
        f"{MAX_RUNS_PER_SESSION} benchmark runs for this session.",
        icon=":material/block:",
    )

run_label = (
    f"Run benchmark — {total_combos} combination{'s' if total_combos != 1 else ''}"
    if ready else "Run benchmark"
)

run_clicked = st.button(
    run_label,
    type="primary",
    disabled=not ready,
    icon=":material/bolt:",
)

if not ready and not run_clicked:
    hints = []

    if not JUDGE_AVAILABLE:
        hints.append(
            "configure a real Groq judge API key in Streamlit Secrets (required for judge)"
        )

    if not task.strip():
        hints.append("enter a task above")

    if len(task.strip()) > MAX_TASK_LENGTH:
        hints.append(f"keep the task under {MAX_TASK_LENGTH} characters")

    if not selected_models:
        hints.append("select at least one model")

    if not selected_strategies:
        hints.append("select at least one strategy")

    if too_many_combinations:
        hints.append(
            f"select no more than {MAX_COMBINATIONS_PER_RUN} combinations"
        )

    if runs_exhausted:
        hints.append(
            f"you have reached the {MAX_RUNS_PER_SESSION}-run demo limit"
        )

    if hints:
        st.info(
            "To run: " + " · ".join(hints) + ".",
            icon=":material/info:",
        )


# ─────────────────────────────────────────────
# Run benchmark
# ─────────────────────────────────────────────
if run_clicked and ready:

    # Final server-side safety check
    total_combos = len(selected_models) * len(selected_strategies)

    if total_combos > MAX_COMBINATIONS_PER_RUN:
        st.error(
            f"This demo allows a maximum of "
            f"{MAX_COMBINATIONS_PER_RUN} combinations per run."
        )
        st.stop()

    if st.session_state.benchmark_runs >= MAX_RUNS_PER_SESSION:
        st.error(
            "You have reached the maximum number of demo runs "
            "for this session."
        )
        st.stop()

    if len(task.strip()) > MAX_TASK_LENGTH:
        st.error(
            f"Task must be {MAX_TASK_LENGTH} characters or fewer."
        )
        st.stop()

    st.header("Results", anchor=False)

    # Column headers for strategies
    col_headers = [STRATEGY_META[s]["label"] for s in selected_strategies]
    if len(selected_strategies) > 1:
        header_cols = st.columns(len(selected_strategies))
        for i, lbl in enumerate(col_headers):
            meta = STRATEGY_META[selected_strategies[i]]
            header_cols[i].markdown(
                f"<div style='text-align:center;font-weight:700;font-size:13px;color:{meta['color']}'>"
                f"{lbl}</div>",
                unsafe_allow_html=True,
            )

    all_scores: dict = {}

    successful_results = 0

    for model_key in selected_models:
        model_meta = MODEL_META[model_key]
        model_api_key = model_meta["api_key"]

        if len(selected_models) > 1:
            st.markdown(
                f"<div style='font-size:12px;font-weight:700;color:#888;margin:12px 0 4px;"
                f"text-transform:uppercase;letter-spacing:.08em'>"
                f"{model_meta['label']}</div>",
                unsafe_allow_html=True,
            )

        result_cols = st.columns(len(selected_strategies))

        for col_idx, strategy_key in enumerate(selected_strategies):
            strat_meta = STRATEGY_META[strategy_key]
            combo_key = f"{model_key}__{strategy_key}"

            with result_cols[col_idx]:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='font-size:12px;font-weight:700;color:{strat_meta['color']}'>"
                        f"{strat_meta['label']}</div>"
                        f"<div style='font-size:11px;color:#888;margin-bottom:6px'>{model_meta['label']}</div>",
                        unsafe_allow_html=True,
                    )

                    # Call model
                    with st.spinner("Calling model…"):
                        t0 = time.time()
                        try:
                            system_p, user_p = build_prompt(strategy_key, task)
                            caller = MODEL_CALLERS[model_key]
                            output = caller(system_p, user_p, model_api_key)
                            latency = round(time.time() - t0, 1)
                            tokens = estimate_tokens(output)
                            error = None
                        except APIError as e:
                            output = str(e)
                            latency = round(time.time() - t0, 1)
                            tokens = 0
                            error = str(e)
                        except Exception as e:
                            output = _requests_error_message(e)  # don't hardcode the generic string
                        except requests.HTTPError as exc:
                            st.write(f"DEBUG: HTTP {exc.response.status_code} — {exc.response.text[:200]}")
                    # Show output
                    st.markdown(
                        f"<div class='output-box'>{output}</div>",
                        unsafe_allow_html=True,
                    )

                    # Judge
                    if not error:
                        with st.spinner("Scoring…"):
                            scores = judge_output(task, strat_meta["label"], output)
                            if scores.get("verdict") == "Judge scoring failed; using fallback scores.":
                                st.warning(
                                    "Judge scoring failed; fallback scores were used.",
                                    icon=":material/warning:",
                                )

                        all_scores[combo_key] = {
                            "model_key": model_key,
                            "strategy_key": strategy_key,
                            "output": output,
                            "scores": scores,
                            "latency": latency,
                            "tokens": tokens,
                        }

                        # Score pill
                        overall = scores.get("overall", 0)
                        oc = SCORE_COLOR.get(int(overall), "#888")
                        verdict = scores.get("verdict", "")
                        st.markdown(
                            f"<div style='margin-top:8px'>"
                            f"<span class='score-pill' style='background:{oc}22;color:{oc}'>"
                            f"⭐ {overall}/5</span> "
                            f"<span style='font-size:11px;color:#888'>{verdict}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # Dimension bars
                        with st.expander("Dimension scores", expanded=False, icon=":material/bar_chart:"):
                            for dim in DIMENSIONS:
                                score = scores.get(dim, 0)
                                bar_w = int(score / 5 * 100)
                                dc = SCORE_COLOR.get(int(score), "#888")
                                st.markdown(
                                    f"<div style='font-size:11px;color:#aaa;margin-bottom:2px'>{dim}</div>"
                                    f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:5px'>"
                                    f"<div style='flex:1;height:5px;background:rgba(255,255,255,0.08);border-radius:3px'>"
                                    f"<div style='height:100%;width:{bar_w}%;background:{dc};border-radius:3px'></div>"
                                    f"</div><span style='font-size:11px;font-weight:700;color:{dc}'>{score}</span></div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(f"~{tokens} tokens · {latency}s latency")
                    else:
                        st.error(_pretty_error(error), icon=":material/error:")

    if all_scores:
        successful_results = len(all_scores)

    if successful_results > 0:
        st.session_state.benchmark_runs += 1

    # ── Summary section ─────────────────────
    if all_scores:
        st.header("Comparison overview", anchor=False)

        tab_chart, tab_table = st.tabs([
            ":material/bar_chart: Quality chart",
            ":material/table: Score table",
        ])

        with tab_chart:
            render_quality_chart(all_scores)

        with tab_table:
            # Build table HTML
            rows_html = ""
            for combo_key, info in all_scores.items():
                m = MODEL_META[info["model_key"]]
                s = STRATEGY_META[info["strategy_key"]]
                sc = info["scores"]
                overall = sc.get("overall", 0)
                oc = SCORE_COLOR.get(int(round(float(overall))), "#888")

                dim_cells = ""
                for dim in DIMENSIONS:
                    v = sc.get(dim, 0)
                    dc = SCORE_COLOR.get(int(v), "#888")
                    dim_cells += (
                        f"<td><span class='score-pill' style='background:{dc}22;color:{dc}'>"
                        f"{v}</span></td>"
                    )

                rows_html += (
                    f"<tr>"
                    f"<td>"
                    f"<div style='font-weight:700;font-size:12px'>{m['label']}</div>"
                    f"<div style='font-size:11px;color:{s['color']};font-weight:600'>{s['label']}</div>"
                    f"</td>"
                    f"{dim_cells}"
                    f"<td><span class='score-pill' style='background:{oc}22;color:{oc};font-size:13px'>{overall:.2f}</span></td>"
                    f"<td style='color:#888;font-size:11px'>{info['tokens']} tok<br>{info['latency']}s</td>"
                    f"</tr>"
                )

            dim_headers = "".join(f"<th>{d}</th>" for d in DIMENSIONS)
            table_html = (
                f"<table class='bench-table'>"
                f"<thead><tr>"
                f"<th style='text-align:left'>Model / Strategy</th>"
                f"{dim_headers}"
                f"<th>Overall</th>"
                f"<th>Tokens / Latency</th>"
                f"</tr></thead>"
                f"<tbody>{rows_html}</tbody>"
                f"</table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

        # Best performer callout
        best_key = max(all_scores, key=lambda k: all_scores[k]["scores"].get("overall", 0))
        best = all_scores[best_key]
        bm = MODEL_META[best["model_key"]]
        bs = STRATEGY_META[best["strategy_key"]]
        st.success(
            f"**Top performer:** {bm['label']} with {bs['label']} "
            f"— Overall score {best['scores'].get('overall', '?')}/5",
            icon=":material/emoji_events:",
        )


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.caption(
    "Week 5 project — AI/LLM Engineer 12-Week Roadmap · "
    f"Judge: {GROQ_JUDGE_MODEL} · All models free-tier · "
    f"Built {datetime.now().strftime('%Y-%m-%d')}"
)
