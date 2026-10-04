from __future__ import annotations

from medrag.audit import SessionCostTracker, log_audit, log_usage
from medrag.privacy import cloud_gate_error, scan_for_phi
from medrag.providers.base import Usage


def test_scan_for_phi_detects_ssn():
    assert "SSN-like pattern" in scan_for_phi("Patient SSN is 123-45-6789")


def test_scan_for_phi_detects_email():
    assert "email address" in scan_for_phi("contact patient at john@example.com")


def test_scan_for_phi_detects_phone():
    assert "phone number" in scan_for_phi("call them at 555-123-4567")


def test_scan_for_phi_detects_mrn_label():
    assert "MRN/DOB-style label" in scan_for_phi("MRN: 00123456 presented with chest pain")


def test_scan_for_phi_detects_dob_near_born():
    assert "date near 'born'/'DOB'" in scan_for_phi("patient born 01/02/1980 presented")


def test_scan_for_phi_clean_question_has_no_hits():
    assert scan_for_phi("What does the evidence say about SGLT2 inhibitors in heart failure?") == []


def test_cloud_gate_blocks_without_env(monkeypatch):
    monkeypatch.delenv("MEDRAG_ALLOW_CLOUD", raising=False)
    assert cloud_gate_error() is not None


def test_cloud_gate_allows_with_env(monkeypatch):
    monkeypatch.setenv("MEDRAG_ALLOW_CLOUD", "1")
    assert cloud_gate_error() is None


def test_session_cost_tracker_under_limit():
    tracker = SessionCostTracker(max_usd=1.0)
    tracker.add(0.5)
    assert tracker.check_exceeded() is None


def test_session_cost_tracker_over_limit():
    tracker = SessionCostTracker(max_usd=1.0)
    tracker.add(0.6)
    tracker.add(0.6)
    err = tracker.check_exceeded()
    assert err is not None
    assert "exceeds MEDRAG_MAX_SESSION_USD" in err


def test_session_cost_tracker_no_limit_never_exceeds():
    tracker = SessionCostTracker(max_usd=None)
    tracker.add(1000.0)
    assert tracker.check_exceeded() is None


def test_log_audit_and_usage_write_jsonl(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    usage_path = tmp_path / "usage.jsonl"
    usage = Usage(input_tokens=100, output_tokens=20)

    log_audit(audit_path, "anthropic", "claude-opus-5-5", "a question", ["111", "222"], usage)
    log_usage(usage_path, "anthropic", "claude-opus-5-5", usage, 0.01)

    assert audit_path.exists()
    assert usage_path.exists()
    audit_line = audit_path.read_text().strip()
    assert "111" in audit_line and "222" in audit_line
    assert "a question" in audit_line
