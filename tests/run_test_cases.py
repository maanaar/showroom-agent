# -*- coding: utf-8 -*-
"""
Manual test runner for showroom-agent test cases.

Run from the showroom-agent directory:
    python tests/run_test_cases.py          # all tests
    python tests/run_test_cases.py --json   # include raw JSON blocks
    python tests/run_test_cases.py --only installment  # filter by label keyword
"""
import sys
import time
import json
import pathlib
import textwrap


sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# ── Colour helpers (Windows-safe) ─────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def _c(text, *codes):
    return "".join(codes) + str(text) + RESET


# ── Helpers ───────────────────────────────────────────────────────────────────


def _divider(char="─", width=70):
    print(_c(char * width, DIM))


def _header(n: int, msg: str):
    print()
    print(_c("═" * 70, CYAN))
    print(_c(f"  TEST {n:02d}", BOLD, CYAN))
    print(_c("═" * 70, CYAN))
    print(_c("  USER ▶ ", BOLD) + msg)


def _print_json_block(label: str, obj):
    print(_c(f"\n  ┌─ {label} ", DIM))
    raw = json.dumps(obj, ensure_ascii=False, indent=4)
    for line in raw.splitlines():
        print(_c("  │ ", DIM) + line)
    print(_c("  └" + "─" * 50, DIM))


def _print_intent(intent: str, product_type: "str | None"):
    colour = GREEN if intent not in ("other", None) else YELLOW
    pt = f" | product_type: {product_type}" if product_type else ""
    print(_c(f"\n  [INTENT]  {intent}{pt}", colour, BOLD))


def _print_vehicles(vehicles: list):
    if not vehicles:
        print(_c("  [VEHICLES] (none)", DIM))
        return
    print(_c(f"\n  [VEHICLES] {len(vehicles)} found:", BOLD))
    for i, v in enumerate(vehicles, 1):
        name = v.get("name_ar") or v.get("name_en") or "?"
        # Custom installment calculation result
        if v.get("monthly_payment") is not None and v.get("months"):
            months  = v.get("months")
            monthly = v.get("monthly_payment", 0)
            total   = v.get("total_repayment", 0)
            rate    = v.get("interest_rate_pct", 0)
            print(
                f"    {i}. {name}  —  {months}ش"
                f"  |  فائدة: {rate}%"
                f"  |  قسط/ش: {int(monthly):,} ج"
                f"  |  إجمالي: {int(total):,} ج"
            )
        else:
            price = v.get("price")
            price_str = f"{int(price):,} ج" if price and str(price) != "None" else "—"
            inst = v.get("installment_12")
            inst_str = f"  |  قسط 12ش: {int(inst):,} ج/ش" if inst and str(inst) != "None" else ""
            print(f"    {i}. {name}  —  {price_str}{inst_str}")


def _print_usage(usage: dict):
    if not usage:
        print(_c("  [TOKENS]  (not reported)", DIM))
        return
    inp      = usage.get("input_tokens",   0)
    out      = usage.get("output_tokens",  0)
    tot      = usage.get("total_tokens",   inp + out)
    thinking = usage.get("thinking_tokens", 0)
    think_str = f"  thinking={thinking}" if thinking else ""
    print(_c(f"\n  [TOKENS]  in={inp}  out={out}  total={tot}{think_str}", CYAN))


def _print_response(text: str):
    print(_c("\n  [RESPONSE]", BOLD, GREEN))
    wrapped = textwrap.fill(text, width=65, initial_indent="    ", subsequent_indent="    ")
    print(wrapped)


def _print_raw_json(body: dict):
    print(_c("\n  [RAW JSON]", DIM))
    raw = json.dumps(body, ensure_ascii=False, indent=4)
    for line in raw.splitlines():
        print(_c("  ", DIM) + line)


def _reset(uid: str):
    client.delete(f"/api/chat/{uid}")


def _send(uid: str, message: str) -> dict:
    r = client.post("/api/chat", json={"user_id": uid, "message": message})
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "tc01",
        "label": "أحدث اسكوتر",
        "message": "احكيلي عن آخر إسكوتر نازل عندكم؟",
        "expected_intent": "browse",
    },
    {
        "id": "tc02",
        "label": "زيت Rebsol",
        "message": "هل عندكم زيت Rebsol للسكوتير Sym sr؟",
        "expected_intent": "other",
    },
    {
        "id": "tc03",
        "label": "اسكوتر للسن الصغير",
        "message": "عايز أختار اسكوتر مناسب للسن الصغير؟",
        "expected_intent": "filter",
    },
    {
        "id": "tc04",
        "label": "أنظمة التقسيط",
        "message": "ايه أنظمة التقسيط المتاحة عندكو؟",
        "expected_intent": "installment",
    },
    {
        "id": "tc05",
        "label": "40k مقدم + 7 شهور SYM ST",
        "message": "دلوقتي انا معايا أربعين الف جنيه و عايز اشتري sym st اقسط ازاي"
                   " ولو دفعتهم مقدم واقسط الباقي علي ٧ شهور هدفع كام وبفايدة قد ايه؟",
        "expected_intent": "installment",
    },
    {
        "id": "tc06",
        "label": "فئات الأسعار",
        "message": "ايه فئات الأسعار للاسكوترات عندكو؟",
        "expected_intent": "browse",
    },
    {
        "id": "tc07",
        "label": "مقارنة jet14 vs zontes",
        "message": "اعملي مقارنة بين sym jet14 و zontes e368؟",
        "expected_intent": "compare",
    },
    {
        "id": "tc08",
        "label": "ساعات العمل",
        "message": "فاتحين امتي؟",
        "expected_intent": "other",
    },
    {
        "id": "tc09",
        "label": "عروض حالية",
        "message": "هو في عروض دلوقتي؟",
        "expected_intent": "other",
    },
    {
        "id": "tc10",
        "label": "أقرب فرع",
        "message": "فين اقرب فرع ليا؟",
        "expected_intent": "other",
    },
    {
        "id": "tc11",
        "label": "تقسيط الزيت",
        "message": "هو ممكن اقسط الزيت يعني ايه حدود التقسيط؟",
        "expected_intent": "installment",
    },
    {
        "id": "tc12",
        "label": "jet x مميزات وعيوب",
        "message": "كلمني عن jet x ايه مميزاته و عيوبه؟",
        "expected_intent": "details",
    },
    {
        "id": "tc13",
        "label": "زيت للسكوتر المشترى",
        "message": "انا اشتريت سكوتر منكم من شهر عايز الزيت المناسب ليه؟",
        "expected_intent": "other",
    },

    # ── Custom installment calculation ────────────────────────────────────────
    {
        "id": "tc14",
        "label": "تقسيط Jet X على 9 شهور",
        "message": "عايز أقسط Jet X على 9 شهور — المقدم كام والقسط الشهري هيبقى كام؟",
        "expected_intent": "installment",
    },
    {
        "id": "tc15",
        "label": "تقسيط Jet X على 12 شهر",
        "message": "لو أقسط Jet X على 12 شهر بدفع كام في الشهر؟",
        "expected_intent": "installment",
    },
    {
        "id": "tc16",
        "label": "تقسيط SYM SR على 36 شهر",
        "message": "هل ينفع أقسط SYM SR 200 على 36 شهر؟ وإيه اللي هدفعه؟",
        "expected_intent": "installment",
    },
    {
        "id": "tc17",
        "label": "تقسيط موتو على 18 شهر",
        "message": "عايز أعرف تقسيط هاوجي k4 على 18 شهر إيه المقدم والقسط الشهري؟",
        "expected_intent": "installment",
    },
    {
        "id": "tc18",
        "label": "ميزانية قسط شهري موتو",
        "message": "عندي ألفين و٥٠٠ قسط شهري — ايه الموتوسيكلات اللي تناسبني على 12 شهر؟",
        "expected_intent": "installment",
    },

    # ── Booking + complaint ───────────────────────────────────────────────────
    {
        "id": "tc19",
        "label": "حجز موعد",
        "message": "عايز أحجز موعد لأجرب الـ Jet X — اسمي محمد وتليفوني 01012345678",
        "expected_intent": "booking",
    },
    {
        "id": "tc20",
        "label": "شكوى خدمة",
        "message": "اشتريت موتوسيكل من عندكم وبعد أسبوع الموتور بدأ يعمل صوت غريب — عايز حل",
        "expected_intent": "complaint",
    },

    # ── Helmet ────────────────────────────────────────────────────────────────
    {
        "id": "tc21",
        "label": "تصفح خوذات",
        "message": "عندكم خوذات ايه؟",
        "expected_intent": "browse",
    },
    {
        "id": "tc22",
        "label": "خوذة رخيصة",
        "message": "عايز خوذة رخيصة تحت ٥٠٠ جنيه",
        "expected_intent": "filter",
    },

    # ── Multi-turn: follow-up on previous context ─────────────────────────────
    {
        "id": "tc23a",
        "label": "تصفح موتو (جلسة متعددة)",
        "message": "عندكم ايه من موتوسيكلات؟",
        "expected_intent": "browse",
        "session": "multi_turn_01",
    },
    {
        "id": "tc23b",
        "label": "تفاصيل بعد التصفح",
        "message": "كلمني أكتر عن الأرخص واحدة منهم",
        "expected_intent": "details",
        "session": "multi_turn_01",   # same session → has context of tc23a
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all(show_raw_json: bool = False, only: str = None):
    passed = 0
    failed = 0
    results = []

    # Filter by label keyword if --only was passed
    cases = TEST_CASES
    if only:
        cases = [tc for tc in TEST_CASES if only.lower() in tc["label"].lower()
                 or only.lower() in tc.get("id", "").lower()]
        if not cases:
            print(_c(f"  No test cases matching --only '{only}'", YELLOW))
            return

    # Track which sessions are fresh (multi-turn cases share a session_id)
    seen_sessions: set = set()

    for n, tc in enumerate(cases, 1):
        # Use explicit session key if provided, else isolate by test id
        uid = tc.get("session") or tc["id"]
        if uid not in seen_sessions:
            _reset(uid)
            seen_sessions.add(uid)
        _header(n, tc["message"])

        t0 = time.time()
        try:
            body = _send(uid, tc["message"])
            elapsed = time.time() - t0

            intent        = body.get("intent") or "—"
            product_type  = None  # not returned by API; intent covers it
            vehicles      = body.get("vehicles", [])
            response_text = body.get("response", "")
            usage         = body.get("usage", {})

            _print_intent(intent, product_type)
            _print_vehicles(vehicles)
            _print_response(response_text)
            _print_usage(usage)

            if show_raw_json:
                _print_raw_json(body)

            # Soft check: intent matches expectation
            expected = tc.get("expected_intent")
            intent_ok = (intent == expected) or (expected is None)
            response_ok = bool(response_text.strip())
            total_tok = usage.get("total_tokens", 0)

            status_line = (
                _c(f"  ✔ PASS", GREEN, BOLD)
                if intent_ok and response_ok
                else _c(f"  ✘ WARN  (expected intent={expected}, got {intent})", YELLOW, BOLD)
            )
            print(f"\n{status_line}   [{elapsed:.1f}s]  tokens={total_tok or '?'}")

            if intent_ok and response_ok:
                passed += 1
            else:
                failed += 1

            results.append({
                "n": n,
                "label": tc["label"],
                "intent": intent,
                "expected": expected,
                "vehicles": len(vehicles),
                "has_response": response_ok,
                "elapsed_s": round(elapsed, 1),
                "tokens": total_tok,
            })

        except Exception as exc:
            elapsed = time.time() - t0
            print(_c(f"\n  ✘ ERROR: {exc}", RED, BOLD))
            failed += 1
            results.append({
                "n": n,
                "label": tc["label"],
                "error": str(exc),
                "elapsed_s": round(elapsed, 1),
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print(_c("═" * 70, CYAN))
    print(_c("  SUMMARY", BOLD, CYAN))
    print(_c("═" * 70, CYAN))
    print(f"  {'#':<4} {'Label':<26} {'Intent':<13} {'Exp':<13} {'Veh':>3} {'Tok':>5} {'Time':>6}")
    _divider()
    for r in results:
        err = r.get("error")
        if err:
            row = f"  {r['n']:<4} {r['label']:<26} {'ERROR':<13} {'—':<13} {'—':>3} {'—':>5} {r['elapsed_s']:>5}s"
            print(_c(row, RED))
        else:
            match = r["intent"] == r["expected"]
            colour = GREEN if match and r["has_response"] else YELLOW
            tok_str = str(r.get("tokens") or "?")
            row = (
                f"  {r['n']:<4} {r['label']:<26} {r['intent']:<13}"
                f" {r['expected']:<13} {r['vehicles']:>3} {tok_str:>5} {r['elapsed_s']:>5}s"
            )
            print(_c(row, colour))

    _divider()
    total = passed + failed
    total_tokens = sum(r.get("tokens") or 0 for r in results if not r.get("error"))
    print(f"\n  {_c(passed, GREEN, BOLD)} / {total} passed   "
          f"{_c(failed, RED, BOLD)} / {total} with warnings/errors   "
          f"total tokens: {_c(total_tokens, CYAN)}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    show_raw  = "--json"  in sys.argv
    only_kw   = next((sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--only" and i + 1 < len(sys.argv)), None)
    run_all(show_raw_json=show_raw, only=only_kw)
