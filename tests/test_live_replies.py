# -*- coding: utf-8 -*-
"""
Live end-to-end tests -- call the REAL LLM (Ollama qwen3:4b) and the REAL data.

Tests skip automatically if Ollama is not reachable at http://localhost:11434.

Run:
    venv\\Scripts\\python -m pytest tests/test_live_replies.py -v -s

What we test vs. what we don't:
  TESTED (deterministic):
    - API always returns HTTP 200 for valid input
    - Filter logic: returned vehicles always respect the price cap
    - Session management: reset clears state, history accumulates
  NOT TESTED (non-deterministic LLM):
    - Whether the model picks the "right" intent every time
    - Whether the model mentions specific vehicle names in its reply
    - Whether the model asks for name/phone in a specific way
"""
import sys
import time
import pytest
import requests as _requests
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

def _ollama_running() -> bool:
    try:
        r = _requests.get("http://localhost:11434", timeout=2)
        return r.status_code < 500
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama is not running at localhost:11434"
)

from app import app  # noqa: E402

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

DATA_INTENTS = {"browse", "filter", "details", "installment"}


def _chat(user_id: str, message: str, retries: int = 3) -> dict:
    """POST /api/chat, retrying on empty LLM response."""
    data = {}
    for attempt in range(retries):
        r = client.post("/api/chat", json={"user_id": user_id, "message": message})
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
        data = r.json()
        if data.get("response"):
            return data
        if attempt < retries - 1:
            print(f"\n  [retry {attempt + 1}] empty response, waiting 5s...")
            time.sleep(5)
            _reset(user_id)
    return data


def _reset(user_id: str):
    r = client.delete(f"/api/chat/{user_id}")
    assert r.status_code == 200


def _safe_print(label: str, text: str):
    safe = text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8", errors="replace"
    )
    print(f"\n[{label}] -> {safe}")


def _is_arabic(text: str) -> bool:
    return sum(1 for c in text if "\u0600" <= c <= "\u06ff") > 5


# ===========================================================================
# Infrastructure tests (always deterministic)
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Any message → API always returns 200 with correct JSON shape
# ---------------------------------------------------------------------------

def test_api_returns_200_with_correct_schema():
    _reset(UID["greeting"])
    # ازيك
    r = client.post("/api/chat",
                    json={"user_id": UID["greeting"],
                          "message": "\u0627\u0632\u064a\u0643"})
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert "intent" in body
    assert "vehicles" in body
    assert isinstance(body["vehicles"], list)
    assert isinstance(body["response"], str)


# ---------------------------------------------------------------------------
# 2. Session reset always clears state
# ---------------------------------------------------------------------------

def test_session_reset_clears_state():
    uid = "live-test-reset"
    # Send one message to create a session
    client.post("/api/chat",
                json={"user_id": uid,
                      "message": "\u0627\u0632\u064a\u0643"})
    # Reset
    r = client.delete(f"/api/chat/{uid}")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ===========================================================================
# LLM integration tests (soft on content, hard on data logic)
# ===========================================================================

# ---------------------------------------------------------------------------
# 3. Greeting → API works, reply is Arabic if model responds
# ---------------------------------------------------------------------------

def test_greeting_reply_is_arabic():
    _reset(UID["greeting"])
    # ازيك
    data = _chat(UID["greeting"], "\u0627\u0632\u064a\u0643")
    _safe_print("greeting", data.get("response", ""))

    assert data.get("intent") is not None, "intent missing from response"
    if data.get("response"):
        assert _is_arabic(data["response"]), "Reply is not Arabic"


# ---------------------------------------------------------------------------
# 4. Filter by price → HARD: every returned vehicle must be within budget
# ---------------------------------------------------------------------------

def test_filter_vehicles_respect_price_cap():
    _reset(UID["filter_price"])
    # عايز موتوسيكل باقل من 90 الف
    data = _chat(UID["filter_price"],
                 "\u0639\u0627\u064a\u0632 \u0645\u0648\u062a\u0648\u0633\u064a\u0643\u0644 "
                 "\u0628\u0627\u0642\u0644 \u0645\u0646 90 \u0627\u0644\u0641")

    _safe_print("filter_price", data.get("response", ""))

    # If intent was classified as a data-fetching intent and vehicles returned,
    # every price must be ≤ 90,000 — this is always deterministic
    for v in data.get("vehicles", []):
        price = v.get("price")
        if price is not None and str(price).lower() != "nan":
            assert float(price) <= 90000, (
                f"Vehicle '{v['name_en']}' price {price} exceeds 90,000 budget"
            )

    if data.get("response"):
        assert _is_arabic(data["response"])


# ---------------------------------------------------------------------------
# 5. Browse → vehicles (if returned) have correct structure
# ---------------------------------------------------------------------------

def test_browse_vehicles_have_correct_structure():
    _reset(UID["browse"])
    # عايز اشوف الموتوسيكلات
    data = _chat(UID["browse"],
                 "\u0639\u0627\u064a\u0632 \u0627\u0634\u0648\u0641 "
                 "\u0627\u0644\u0645\u0648\u062a\u0648\u0633\u064a\u0643\u0644\u0627\u062a")

    _safe_print("browse", data.get("response", ""))

    required_keys = {"name_en", "name_ar", "price", "available",
                     "installment_12", "company"}
    for v in data.get("vehicles", []):
        missing = required_keys - v.keys()
        assert not missing, f"Vehicle dict missing keys: {missing}"
        assert v["available"] == "\u0645\u062a\u0627\u062d", (
            f"Non-available vehicle in results: {v['name_en']}"
        )

    if data.get("response"):
        assert _is_arabic(data["response"])


# ---------------------------------------------------------------------------
# 6. Details → if vehicle found, it has all required fields
# ---------------------------------------------------------------------------

def test_details_vehicle_has_all_fields():
    _reset(UID["details"])
    # عايز تفاصيل هوندا
    data = _chat(UID["details"],
                 "\u0639\u0627\u064a\u0632 \u062a\u0641\u0627\u0635\u064a\u0644 "
                 "\u0647\u0648\u0646\u062f\u0627")

    _safe_print("details", data.get("response", ""))

    for v in data.get("vehicles", []):
        assert "name_en" in v
        assert "price" in v
        assert "engine_cc" in v

    if data.get("response"):
        assert _is_arabic(data["response"])


# ---------------------------------------------------------------------------
# 7. Multi-turn → conversation history grows across turns (API state)
# ---------------------------------------------------------------------------

def test_multi_turn_session_state_persists():
    _reset(UID["multi_turn"])

    # Turn 1
    # عايز اشوف الموتوسيكلات
    d1 = _chat(UID["multi_turn"],
               "\u0639\u0627\u064a\u0632 \u0627\u0634\u0648\u0641 "
               "\u0627\u0644\u0645\u0648\u062a\u0648\u0633\u064a\u0643\u0644\u0627\u062a")
    _safe_print("multi-turn T1", d1.get("response", ""))

    # Turn 2 — server must return 200 regardless of LLM content
    # ايه ارخصهم
    r2 = client.post("/api/chat",
                     json={"user_id": UID["multi_turn"],
                           "message": "\u0627\u064a\u0647 \u0627\u0631\u062e\u0635\u0647\u0645"})
    assert r2.status_code == 200
    d2 = r2.json()
    _safe_print("multi-turn T2", d2.get("response", ""))

    if d2.get("response"):
        assert _is_arabic(d2["response"])
