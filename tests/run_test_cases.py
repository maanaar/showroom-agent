# -*- coding: utf-8 -*-
"""
Manual test runner for the 13 showroom-agent test cases.

Run from the showroom-agent directory:
    python tests/run_test_cases.py

Requires Ollama running at http://localhost:11434 with qwen3:4b pulled.
"""
import sys
import time
import json
import pathlib
import textwrap
import requests as _requests

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

def _ollama_running() -> bool:
    try:
        return _requests.get("http://localhost:11434", timeout=2).status_code < 500
    except Exception:
        return False


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


def _print_intent(intent: str, product_type: str | None):
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
        price = v.get("price")
        price_str = f"{int(price):,} ج" if price and str(price) != "None" else "—"
        inst = v.get("installment_12")
        inst_str = f"  |  قسط 12ش: {int(inst):,} ج/ش" if inst and str(inst) != "None" else ""
        print(f"    {i}. {name}  —  {price_str}{inst_str}")


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
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all(show_raw_json: bool = False):
    passed = 0
    failed = 0
    results = []

    for n, tc in enumerate(TEST_CASES, 1):
        uid = tc["id"]
        _reset(uid)
        _header(n, tc["message"])

        t0 = time.time()
        try:
            body = _send(uid, tc["message"])
            elapsed = time.time() - t0

            intent        = body.get("intent") or "—"
            product_type  = None  # not returned by API; intent covers it
            vehicles      = body.get("vehicles", [])
            response_text = body.get("response", "")

            _print_intent(intent, product_type)
            _print_vehicles(vehicles)
            _print_response(response_text)

            if show_raw_json:
                _print_raw_json(body)

            # Soft check: intent matches expectation
            expected = tc.get("expected_intent")
            intent_ok = (intent == expected) or (expected is None)
            response_ok = bool(response_text.strip())

            status_line = (
                _c(f"  ✔ PASS", GREEN, BOLD)
                if intent_ok and response_ok
                else _c(f"  ✘ WARN  (expected intent={expected}, got {intent})", YELLOW, BOLD)
            )
            print(f"\n{status_line}   [{elapsed:.1f}s]")

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
    print(f"  {'#':<4} {'Label':<28} {'Intent':<14} {'Exp':<14} {'Veh':>4} {'Time':>6}")
    _divider()
    for r in results:
        err = r.get("error")
        if err:
            row = f"  {r['n']:<4} {r['label']:<28} {'ERROR':<14} {'—':<14} {'—':>4} {r['elapsed_s']:>5}s"
            print(_c(row, RED))
        else:
            match = r["intent"] == r["expected"]
            colour = GREEN if match and r["has_response"] else YELLOW
            row = (
                f"  {r['n']:<4} {r['label']:<28} {r['intent']:<14}"
                f" {r['expected']:<14} {r['vehicles']:>4} {r['elapsed_s']:>5}s"
            )
            print(_c(row, colour))

    _divider()
    total = passed + failed
    print(f"\n  {_c(passed, GREEN, BOLD)} / {total} passed   "
          f"{_c(failed, RED, BOLD)} / {total} with warnings/errors\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _ollama_running():
        print(_c("✘ Ollama is not running at http://localhost:11434 — aborting.", RED, BOLD))
        sys.exit(1)

    show_raw = "--json" in sys.argv
    run_all(show_raw_json=show_raw)
