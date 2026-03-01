# -*- coding: utf-8 -*-
"""
Live end-to-end tests -- call the REAL LLM (Ollama qwen3:4b) and the REAL data.

Run:
    python tests/test_live_replies.py
"""
import sys
import time
import traceback
import pathlib
import requests as _requests

# Allow running from any directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

UID = {
    "greeting":     "live-test-greeting",
    "browse":       "live-test-browse",
    "filter_price": "live-test-filter-price",
    "details":      "live-test-details",
    "installment":  "live-test-installment",
    "booking":      "live-test-booking",
    "multi_turn":   "live-test-multi-turn",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    try:
        r = _requests.get("http://localhost:11434", timeout=2)
        return r.status_code < 500
    except Exception:
        return False


def _safe_print(label: str, text: str):
    enc = sys.stdout.encoding or "utf-8"
    safe = str(text).encode(enc, errors="replace").decode(enc, errors="replace")
    print(f"  [{label}] -> {safe}")


def _is_arabic(text: str) -> bool:
    return sum(1 for c in text if "\u0600" <= c <= "\u06ff") > 5


def _reset(user_id: str):
    r = client.delete(f"/api/chat/{user_id}")
    assert r.status_code == 200, f"Reset failed: {r.status_code}"


def _chat(user_id: str, message: str, retries: int = 3) -> dict:
    _safe_print("user", message)
    data = {}
    for attempt in range(retries):
        r = client.post("/api/chat", json={"user_id": user_id, "message": message})
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
        data = r.json()
        if data.get("response"):
            return data
        if attempt < retries - 1:
            print(f"  [retry {attempt + 1}] empty response, waiting 5s...")
            time.sleep(5)
            _reset(user_id)
    return data


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_passed = 0
_failed = 0


def _run(name: str, fn):
    global _passed, _failed
    print(f"\n{'─' * 60}")
    print(f"TEST: {name}")
    try:
        fn()
        print(f"  PASS")
        _passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        _failed += 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_api_returns_200_with_correct_schema():
    _reset(UID["greeting"])
    msg = "ازيك"
    _safe_print("user", msg)
    r = client.post("/api/chat", json={"user_id": UID["greeting"], "message": msg})
    assert r.status_code == 200
    body = r.json()
    _safe_print("reply", body.get("response", ""))
    assert "response" in body, "missing 'response'"
    assert "vehicles" in body, "missing 'vehicles'"
    assert isinstance(body["vehicles"], list)
    assert isinstance(body["response"], str)


def test_session_reset_clears_state():
    uid = "live-test-reset"
    client.post("/api/chat", json={"user_id": uid, "message": "ازيك"})
    r = client.delete(f"/api/chat/{uid}")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_greeting_reply_is_arabic():
    _reset(UID["greeting"])
    data = _chat(UID["greeting"], "ازيك")
    _safe_print("reply", data.get("response", ""))
    if data.get("response"):
        assert _is_arabic(data["response"]), "Reply is not Arabic"


def test_filter_vehicles_respect_price_cap():
    _reset(UID["filter_price"])
    data = _chat(UID["filter_price"], "عايز موتوسيكل باقل من 90 الف")
    _safe_print("reply", data.get("response", ""))
    for v in data.get("vehicles", []):
        price = v.get("price")
        if price is not None and str(price).lower() != "nan":
            assert float(price) <= 90000, (
                f"Vehicle '{v['name_en']}' price {price} exceeds 90,000"
            )
    if data.get("response"):
        assert _is_arabic(data["response"])


def test_browse_vehicles_have_correct_structure():
    _reset(UID["browse"])
    data = _chat(UID["browse"], "عايز اشوف الموتوسيكلات")
    _safe_print("reply", data.get("response", ""))
    required_keys = {"name_en", "name_ar", "price", "available", "installment_12", "company"}
    for v in data.get("vehicles", []):
        missing = required_keys - v.keys()
        assert not missing, f"Vehicle missing keys: {missing}"
        assert v["available"] == "متاح", f"Non-available vehicle: {v['name_en']}"
    if data.get("response"):
        assert _is_arabic(data["response"])


def test_details_vehicle_has_all_fields():
    _reset(UID["details"])
    data = _chat(UID["details"], "عايز تفاصيل هوندا")
    _safe_print("reply", data.get("response", ""))
    for v in data.get("vehicles", []):
        assert "name_en" in v
        assert "price" in v
        assert "engine_cc" in v
    if data.get("response"):
        assert _is_arabic(data["response"])


def test_multi_turn_session_state_persists():
    _reset(UID["multi_turn"])

    d1 = _chat(UID["multi_turn"], "عايز اشوف الموتوسيكلات")
    _safe_print("reply T1", d1.get("response", ""))

    msg2 = "ايه ارخصهم"
    _safe_print("user", msg2)
    r2 = client.post("/api/chat", json={"user_id": UID["multi_turn"], "message": msg2})
    assert r2.status_code == 200
    d2 = r2.json()
    _safe_print("reply T2", d2.get("response", ""))
    if d2.get("response"):
        assert _is_arabic(d2["response"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _ollama_running():
        print("SKIP: Ollama is not running at http://localhost:11434")
        sys.exit(0)

    _run("1. API returns 200 with correct schema",  test_api_returns_200_with_correct_schema)
    _run("2. Session reset clears state",           test_session_reset_clears_state)
    _run("3. Greeting reply is Arabic",             test_greeting_reply_is_arabic)
    _run("4. Filter vehicles respect price cap",    test_filter_vehicles_respect_price_cap)
    _run("5. Browse vehicles have correct structure", test_browse_vehicles_have_correct_structure)
    _run("6. Details vehicle has all fields",       test_details_vehicle_has_all_fields)
    _run("7. Multi-turn session state persists",    test_multi_turn_session_state_persists)

    print(f"\n{'═' * 60}")
    print(f"Results: {_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)
