"""The three-agent crew (sequential process).

  1. Data Extractor      -> tools: market snapshot, fundamentals
  2. Financial Analyst   -> tools: valuation/DCF, annual-report RAG search, decision memory
  3. Portfolio Manager   -> tools: portfolio state, risk-managed trade plan
                            output: strict JSON signal, validated by a guardrail
"""
import json
import re

from crewai import Agent, Crew, Process, Task

from . import config, db
from .tools import crew_tools as T

DOC = config.M["doc_name"]            # "SEC 10-K annual report" | "company annual report"
EX_TAG = ("[TICKER 10-K FY2025 · Risk Factors · #4]" if config.M["corpus"] == "sec"
          else "[TICKER.NS AR FY2026 · Risk Management · p57 #212]")
CUR = config.M["currency"]

SIGNAL_KEYS = {"ticker", "action", "confidence", "time_horizon", "rationale", "key_risks", "citations"}


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def norm_tag(c) -> str:
    """Compare citations by content, not punctuation: models often drop the [brackets] or quotes,
    or add spaces, when they copy a tag into JSON."""
    c = re.sub(r"\s+", " ", str(c)).strip().strip("'\"").strip()
    c = c.strip("[]").strip()
    return f"[{c}]"


def fact_check(sig: dict) -> list:
    """Compare the claims in the PM's rationale with the numbers the tools actually returned."""
    try:
        v = T.valuation(config.normalize_ticker(sig["ticker"]))
        tech = T.tech(config.normalize_ticker(sig["ticker"]))
    except Exception:  # noqa: BLE001 - no data, nothing to check against
        return []
    text = str(sig.get("rationale", "")).lower()
    wrong = []
    score = v["quant_score"]["composite"]
    for claimed in re.findall(r"composite(?: score)?(?: of| is| at|:|=)?\s*(\d{1,3})", text):
        if abs(int(claimed) - score) > 1:
            wrong.append(f"composite score is {score}, not {claimed}")
    mos = (v.get("dcf") or {}).get("margin_of_safety_pct")
    if mos is not None:
        if mos < 0 and re.search(r"positive margin of safety|below (the )?(intrinsic|fair|dcf)|undervalued", text):
            wrong.append(f"the price is {abs(mos)}% ABOVE the base-case DCF value (margin of safety {mos}%)")
        if mos > 0 and re.search(r"negative margin of safety|above (the )?(intrinsic|fair|dcf)|overvalued", text):
            wrong.append(f"the price is {mos}% BELOW the base-case DCF value (margin of safety +{mos}%)")
    trend = tech.get("trend")
    if trend == "downtrend" and re.search(r"\buptrend\b", text):
        wrong.append("the price trend is a downtrend (below the 50- and 200-day averages)")
    if trend == "uptrend" and re.search(r"\bdowntrend\b", text):
        wrong.append("the price trend is an uptrend (above the 50- and 200-day averages)")
    return wrong


def validate_signal(output):
    """CrewAI guardrail: returns (ok, value_or_error). On failure CrewAI feeds the
    error back to the agent and retries - this is our output-level error recovery."""
    ok, value = _check_signal(output)
    if not ok:
        db.log_event("guardrail_retry", "portfolio_manager", value, T.RUN["run_id"])
        T.notify(f"🔁 Guardrail rejected PM output: {value}")
    return ok, value


def _check_signal(output):
    try:
        sig = extract_json(output.raw)
    except Exception as exc:  # noqa: BLE001
        return False, f"Output must be ONE valid JSON object only. Parse error: {exc}"
    missing = SIGNAL_KEYS - sig.keys()
    if missing:
        return False, f"JSON is missing required keys: {sorted(missing)}"
    sig["action"] = str(sig["action"]).upper().strip()
    if sig["action"] not in {"BUY", "HOLD", "SELL"}:
        return False, "action must be exactly BUY, HOLD or SELL"
    try:
        sig["confidence"] = float(sig["confidence"])
    except (TypeError, ValueError):
        return False, "confidence must be a number between 0 and 1"
    if sig["confidence"] > 1:
        sig["confidence"] /= 100
    if not 0 <= sig["confidence"] <= 1:
        return False, "confidence must be between 0 and 1"
    if not isinstance(sig["citations"], list) or not sig["citations"]:
        return False, f"citations must be a non-empty list of the {DOC} citation tags used by the analyst, e.g. '{EX_TAG}'"
    if not isinstance(sig["key_risks"], list):
        return False, "key_risks must be a list of strings"
    real = {norm_tag(c) for c in T.RUN["citations"]}
    if real:
        valid = [norm_tag(c) for c in sig["citations"] if norm_tag(c) in real]
        if not valid:
            return False, ("None of your citations were actually retrieved. Use only these exact tags from "
                           f"search_annual_report: {sorted(real)[:6]}")
        sig["citations"] = valid  # silently drop any invented extras
    wrong = fact_check(sig)
    if wrong:
        return False, "Your rationale contradicts the tool results: " + "; ".join(wrong) + ". Rewrite it with the correct facts."
    if "plan_position" not in T.RUN["called"]:
        return False, ("You answered without calling the plan_position tool. Call get_portfolio_state and "
                       "plan_position first, then copy stop_loss, take_profit and shares from plan_position.")
    return True, json.dumps(sig)


def build_crew(ticker: str, llm, step_callback=None) -> Crew:
    extractor = Agent(
        role="Market Data Extractor",
        goal=f"Collect accurate, current market and fundamental data for {ticker}.",
        backstory="A meticulous data engineer on a quant desk. You only report numbers returned by your tools.",
        tools=T.EXTRACTOR_TOOLS, llm=llm, allow_delegation=False, max_iter=6, verbose=True,
    )
    analyst = Agent(
        role="Senior Equity Research Analyst",
        goal=f"Produce an evidence-based investment view on {ticker} combining valuation maths with {DOC} evidence.",
        backstory=("A CFA charterholder who never states an opinion without a number or a citation. "
                   "You use the valuation tool for all arithmetic and the annual-report search for qualitative evidence. "
                   f"All money figures are in {CUR}."),
        tools=T.ANALYST_TOOLS, llm=llm, allow_delegation=False, max_iter=8, verbose=True,
    )
    pm = Agent(
        role="Portfolio Manager",
        goal=f"Turn the research on {ticker} into one disciplined, risk-managed paper-trading decision.",
        backstory=("You run a long-only paper portfolio with hard risk limits. You prefer HOLD when evidence is mixed, "
                   "and you never size a position by instinct - you use the plan_position tool."),
        tools=T.PM_TOOLS, llm=llm, allow_delegation=False, max_iter=6, verbose=True,
    )

    t_extract = Task(
        description=(
            f"Ticker: {ticker}\n"
            f"1. Call get_market_snapshot with '{ticker}'.\n"
            f"2. Call get_fundamentals with '{ticker}'.\n"
            "Then write a DATA BRIEF with these headings: Company, Price & Trend, Valuation Multiples, "
            "Profitability & Growth, Balance Sheet, Data Source. Use bullet points and ONLY numbers from the tools. "
            "If a tool returned an error, say which data is missing."
        ),
        expected_output="A concise DATA BRIEF (max 200 words) with the six headings and tool-sourced numbers only.",
        agent=extractor,
    )
    t_analyse = Task(
        description=(
            f"Ticker: {ticker}. Using the DATA BRIEF as context:\n"
            f"1. Call run_valuation with '{ticker}'.\n"
            f"2. Call search_annual_report for '{ticker}' with question 'key business risks and competitive threats'.\n"
            f"3. Call search_annual_report for '{ticker}' with question 'revenue growth drivers and management outlook'.\n"
            f"4. Call get_past_decisions with '{ticker}'.\n"
            "Then write an ANALYST REPORT with these sections:\n"
            "- Valuation: P/E, EV/EBITDA, FCF yield, DCF bear/base/bull, margin of safety, and the market-implied FCF growth "
            "(reverse DCF). The DCF is deliberately conservative, so judge valuation mainly by asking: is the market-implied "
            f"growth plausible given the growth evidence in the {DOC}?\n"
            "- Quant score: value / quality / momentum / composite from run_valuation.\n"
            "- Bull case: 2-3 points.\n"
            f"- Bear case / risks: 2-3 points, each ending with the exact citation tag from search_annual_report, e.g. {EX_TAG}.\n"
            "- Risk flags: list the risk_flags from run_valuation.\n"
            "- Memory: how this view compares with past decisions.\n"
            "- Verdict: Attractive / Neutral / Unattractive, with one sentence why. Guide: composite >= 70 with plausible "
            "implied growth -> Attractive; composite < 50 or implausible implied growth with serious risks -> Unattractive.\n"
            "Do not invent numbers or citations."
        ),
        expected_output="An ANALYST REPORT (max 380 words) with the seven sections, real numbers and real citation tags from search_annual_report.",
        agent=analyst,
        context=[t_extract],
    )
    t_decide = Task(
        description=(
            f"Ticker: {ticker}. Using the ANALYST REPORT:\n"
            "You MUST call both tools before answering - an answer without them is rejected:\n"
            "1. Call get_portfolio_state.\n"
            f"2. Call plan_position with '{ticker}'.\n"
            "3. Decide using these fund rules:\n"
            "   - BUY if the analyst verdict is Attractive (composite score >= 70) and plan_position allows shares > 0.\n"
            "   - SELL if we already hold the stock and the verdict is Unattractive (or composite < 50).\n"
            "   - Otherwise HOLD.\n"
            "   Confidence (0-1): start from composite/100, subtract 0.05 for each risk flag, never above 0.9.\n"
            "Return ONLY a JSON object, no prose, with exactly these keys:\n"
            '{"ticker": "...", "action": "BUY|HOLD|SELL", "confidence": 0.0, "time_horizon": "e.g. 3-6 months", '
            '"rationale": "2-3 sentences citing valuation numbers", "key_risks": ["...", "..."], '
            '"citations": ["<exact citation tags returned by search_annual_report>", "..."], '
            '"stop_loss": <number from plan_position>, "take_profit": <number from plan_position>, "shares": <number from plan_position or 0>}'
        ),
        expected_output="One valid JSON object with the keys listed. No markdown, no extra text.",
        agent=pm,
        context=[t_analyse],
        guardrail=validate_signal,
        guardrail_max_retries=3,
    )
    return Crew(
        agents=[extractor, analyst, pm],
        tasks=[t_extract, t_analyse, t_decide],
        process=Process.sequential,
        verbose=True,
        step_callback=step_callback,
    )
