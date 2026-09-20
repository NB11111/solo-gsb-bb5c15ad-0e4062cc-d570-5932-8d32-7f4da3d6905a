"""Tests for POST /api/v1/negotiate/coverage-plan and plan_coverage core."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.negotiation import SampleHeaderError, plan_coverage

client = TestClient(app)

PATH = "/api/v1/negotiate/coverage-plan"


def _post(payload):
    return client.post(PATH, json=payload)


def _sample(header, traffic):
    return {"accept_language": header, "traffic": traffic}


def test_basic_planning_selects_rescuing_candidate():
    response = _post(
        {
            "supported_languages": ["en-US", "zh-CN"],
            "default_language": "en-US",
            "samples": [
                _sample("fr-FR,fr;q=0.9", 100),
                _sample("zh-CN", 300),
                _sample(None, 10),
            ],
            "candidate_languages": ["fr-FR", "de-DE"],
            "max_additions": 2,
        }
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selected_additions"] == ["fr-FR"]
    assert body["baseline_fallback_traffic"] == 110
    assert body["improved_traffic"] == 100
    by_header = {
        s["accept_language"] if s["accept_language"] is not None else "":
        s
        for s in body["samples"]
    }
    french = by_header["fr-FR,fr;q=0.9"]
    assert french["before_selected_language"] == "en-US"
    assert french["before_fallback"] is True
    assert french["after_selected_language"] == "fr-FR"
    assert french["after_fallback"] is False
    assert french["improved"] is True
    zh = by_header["zh-CN"]
    assert zh["before_selected_language"] == "zh-CN"
    assert zh["before_fallback"] is False
    assert zh["after_selected_language"] == "zh-CN"
    assert zh["improved"] is False
    no_header = by_header[""]
    assert no_header["before_fallback"] is True
    assert no_header["after_fallback"] is True
    assert no_header["improved"] is False


def test_combination_improvement_selects_multiple_additions():
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr", 100), _sample("de", 90)],
            "candidate_languages": ["fr-FR", "de-DE"],
            "max_additions": 2,
        }
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selected_additions"] == ["fr-FR", "de-DE"]
    assert body["baseline_fallback_traffic"] == 190
    assert body["improved_traffic"] == 190
    assert [s["improved"] for s in body["samples"]] == [True, True]


def test_max_additions_caps_subset_size():
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr", 100), _sample("de", 90)],
            "candidate_languages": ["fr-FR", "de-DE"],
            "max_additions": 1,
        }
    )
    body = response.json()
    # Only one addition allowed: the higher-traffic rescue wins.
    assert body["selected_additions"] == ["fr-FR"]
    assert body["improved_traffic"] == 100
    assert [s["after_fallback"] for s in body["samples"]] == [False, True]


def test_tie_prefers_fewer_additions_then_earlier_candidate():
    # Both candidates rescue the same single sample via truncation prefix;
    # a one-element subset reaches the same improvement as any larger one,
    # and fr-CA precedes fr-FR in candidate order.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr;q=0.9", 100)],
            "candidate_languages": ["fr-CA", "fr-FR"],
            "max_additions": 2,
        }
    )
    body = response.json()
    assert body["selected_additions"] == ["fr-CA"]
    assert body["improved_traffic"] == 100


def test_earlier_candidate_order_tiebreak_with_equal_single_value():
    # Same size, same traffic, disjoint samples: earliest input order wins.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr", 50), _sample("de", 50)],
            "candidate_languages": ["de-DE", "fr-FR"],
            "max_additions": 1,
        }
    )
    body = response.json()
    assert body["selected_additions"] == ["de-DE"]
    assert body["improved_traffic"] == 50


def test_no_improvement_returns_empty_selection():
    response = _post(
        {
            "supported_languages": ["en-US", "zh-CN"],
            "default_language": "en-US",
            "samples": [_sample("en", 10)],
            "candidate_languages": ["fr-FR"],
            "max_additions": 1,
        }
    )
    body = response.json()
    assert body["selected_additions"] == []
    assert body["baseline_fallback_traffic"] == 0
    assert body["improved_traffic"] == 0
    sample = body["samples"][0]
    assert sample["before_selected_language"] == "en-US"
    assert sample["after_selected_language"] == "en-US"
    assert sample["improved"] is False


def test_wildcard_q_zero_blocks_rescue():
    # A wildcard q=0 zeroes every non-concrete candidate, so no addition
    # can turn the fallback into a normal hit.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("*;q=0", 50)],
            "candidate_languages": ["fr-FR", "de-DE"],
            "max_additions": 2,
        }
    )
    body = response.json()
    assert body["selected_additions"] == []
    assert body["baseline_fallback_traffic"] == 50
    assert body["improved_traffic"] == 0
    assert body["samples"][0]["after_fallback"] is True
    assert body["samples"][0]["improved"] is False


def test_concrete_q_zero_candidate_cannot_be_selected():
    # fr;q=0 excludes any fr-* candidate via truncation prefix, while the
    # positive concrete de range rescues de-DE; the sample falls back only
    # because neither range matches the existing en-US tag.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr;q=0,de;q=0.9", 50)],
            "candidate_languages": ["fr-FR", "de-DE"],
            "max_additions": 2,
        }
    )
    body = response.json()
    assert body["selected_additions"] == ["de-DE"]
    assert body["improved_traffic"] == 50
    assert body["samples"][0]["after_selected_language"] == "de-DE"


def test_zero_traffic_counts_for_selection_but_not_weight():
    # A zero-traffic fallback sample is still marked improved, but adds
    # nothing to improved_traffic; the plan optimizes weighted traffic.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr", 0), _sample("de", 5)],
            "candidate_languages": ["fr-FR", "de-DE"],
            "max_additions": 1,
        }
    )
    body = response.json()
    assert body["selected_additions"] == ["de-DE"]
    assert body["improved_traffic"] == 5
    assert body["samples"][0]["improved"] is False
    assert body["samples"][1]["improved"] is True


def test_prefix_truncation_match_rescues():
    # ja matches ja-JP via RFC 4647 truncation, same semantics as /negotiate.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("ja;q=0.8", 20)],
            "candidate_languages": ["ja-JP"],
            "max_additions": 1,
        }
    )
    body = response.json()
    assert body["selected_additions"] == ["ja-JP"]
    assert body["samples"][0]["after_selected_language"] == "ja-JP"
    assert body["samples"][0]["improved"] is True


def test_wildcard_positive_sample_already_hits_is_not_improved():
    # *;q=0.1 makes existing en-US a normal hit; no fallback to rescue.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("ja;q=0.8,*;q=0.1", 60)],
            "candidate_languages": ["ja-JP"],
            "max_additions": 1,
        }
    )
    body = response.json()
    assert body["selected_additions"] == []
    assert body["baseline_fallback_traffic"] == 0
    assert body["samples"][0]["before_selected_language"] == "en-US"
    assert body["samples"][0]["improved"] is False


def test_additions_appended_preserve_priority_of_existing_languages():
    # Existing supported languages keep their list priority. A second
    # sample forces de-DE into the plan; for the first sample en-US then
    # already matched concrete en;q=0.2 and stayed a normal hit, while the
    # higher-q new de-DE wins the selection after planning — that switch
    # is reported per sample but never counts as fallback rescue.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [
                _sample("de;q=0.9,en;q=0.2", 10),
                _sample("de", 5),
            ],
            "candidate_languages": ["de-DE"],
            "max_additions": 1,
        }
    )
    body = response.json()
    assert body["selected_additions"] == ["de-DE"]
    assert body["baseline_fallback_traffic"] == 5
    assert body["improved_traffic"] == 5
    first, second = body["samples"]
    assert first["before_selected_language"] == "en-US"
    assert first["before_fallback"] is False
    assert first["after_selected_language"] == "de-DE"
    assert first["improved"] is False
    assert second["before_fallback"] is True
    assert second["after_selected_language"] == "de-DE"
    assert second["improved"] is True


def test_invalid_sample_header_422_with_element_loc():
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [
                _sample("fr", 1),
                _sample("de;q=2", 2),
            ],
            "candidate_languages": ["fr-FR"],
            "max_additions": 1,
        }
    )
    assert response.status_code == 422
    detail = response.json()["detail"][0]
    assert detail["loc"] == ["body", "samples", 1, "accept_language"]
    assert detail["type"] == "accept_language.syntax_error"
    assert "第 2 个流量样本" in detail["msg"]


@pytest.mark.parametrize(
    "payload,loc",
    [
        # negative traffic
        (
            {
                "supported_languages": ["en-US"],
                "default_language": "en-US",
                "samples": [_sample("fr", -1)],
                "candidate_languages": ["fr-FR"],
                "max_additions": 1,
            },
            ["body", "samples", 0, "traffic"],
        ),
        # candidate already supported
        (
            {
                "supported_languages": ["en-US"],
                "default_language": "en-US",
                "samples": [_sample("fr", 1)],
                "candidate_languages": ["en-us"],
                "max_additions": 1,
            },
            ["body"],
        ),
        # duplicate candidates (case-insensitive)
        (
            {
                "supported_languages": ["en-US"],
                "default_language": "en-US",
                "samples": [_sample("fr", 1)],
                "candidate_languages": ["fr-FR", "fr-fr"],
                "max_additions": 1,
            },
            ["body", "candidate_languages"],
        ),
        # max_additions out of range
        (
            {
                "supported_languages": ["en-US"],
                "default_language": "en-US",
                "samples": [_sample("fr", 1)],
                "candidate_languages": ["fr-FR"],
                "max_additions": 0,
            },
            ["body", "max_additions"],
        ),
        # default not in supported (legacy rule preserved)
        (
            {
                "supported_languages": ["en-US"],
                "default_language": "fr-FR",
                "samples": [_sample("fr", 1)],
                "candidate_languages": ["de-DE"],
                "max_additions": 1,
            },
            ["body"],
        ),
    ],
)
def test_validation_errors_422(payload, loc):
    response = _post(payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == loc


def test_samples_length_bounds():
    base = {
        "supported_languages": ["en-US"],
        "default_language": "en-US",
        "candidate_languages": ["fr-FR"],
        "max_additions": 1,
    }
    response = _post({**base, "samples": []})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "samples"]


def test_candidates_length_bounds():
    base = {
        "supported_languages": ["en-US"],
        "default_language": "en-US",
        "samples": [_sample("fr", 1)],
        "max_additions": 1,
    }
    response = _post({**base, "candidate_languages": []})
    assert response.status_code == 422
    response = _post({**base, "candidate_languages": [f"x-tag{i}" for i in range(13)]})
    assert response.status_code == 422


def test_extra_field_rejected_422():
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr", 1)],
            "candidate_languages": ["fr-FR"],
            "max_additions": 1,
            "unexpected": True,
        }
    )
    assert response.status_code == 422


def test_max_additions_above_candidate_count_is_capped():
    # 1 candidate but max_additions=3 is valid; the search just caps size.
    response = _post(
        {
            "supported_languages": ["en-US"],
            "default_language": "en-US",
            "samples": [_sample("fr", 7)],
            "candidate_languages": ["fr-FR"],
            "max_additions": 3,
        }
    )
    assert response.status_code == 200
    assert response.json()["selected_additions"] == ["fr-FR"]


def test_existing_endpoints_unchanged():
    # The new feature must not alter the legacy endpoint payload.
    response = client.post(
        "/api/v1/negotiate",
        json={
            "accept_language": "zh-CN,zh;q=0.9,en;q=0.8,*;q=0.5",
            "supported_languages": ["en-US", "zh-CN", "zh-TW", "ja-JP"],
            "default_language": "en-US",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["selected_language"] == "zh-CN"
    assert "locale_fallback" not in body
    assert set(body) == {
        "selected_language",
        "selected_supported_index",
        "fallback",
        "fallback_reason",
        "message",
        "parsed_ranges",
        "ignored_duplicates",
        "candidates",
    }
    root = client.get("/").json()
    assert root["negotiate_coverage_plan"] == "POST " + PATH


# ---- core-level unit tests -------------------------------------------------


def test_plan_coverage_core_returns_report_dicts():
    result = plan_coverage(
        samples=[("fr;q=0.9", 100), (None, 5)],
        supported=["en-US"],
        default="en-US",
        candidates=["fr-FR", "de-DE"],
        max_additions=2,
    )
    assert result["selected_additions"] == ["fr-FR"]
    assert result["baseline_fallback_traffic"] == 105
    assert result["improved_traffic"] == 100
    assert len(result["samples"]) == 2


def test_plan_coverage_core_raises_sample_header_error():
    with pytest.raises(SampleHeaderError) as excinfo:
        plan_coverage(
            samples=[("fr", 1), ("bad;;", 2)],
            supported=["en-US"],
            default="en-US",
            candidates=["fr-FR"],
            max_additions=1,
        )
    assert excinfo.value.position == 2
