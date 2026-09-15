"""Unit tests for Gemma4_Ollama static helpers.

The module imports PyQt6 at the top level, so these tests skip
gracefully on a headless environment without PyQt installed.
"""
import pytest

pytest.importorskip("PyQt6")

from anylabeling.services.auto_labeling.gemma4_ollama import Gemma4_Ollama


def test_build_prompt_with_dot_separated_labels():
    prompt = Gemma4_Ollama._build_prompt("helmet . person . fire")
    assert "helmet, person, fire" in prompt
    assert "box_2d" in prompt
    assert "STRICT JSON" in prompt


def test_build_prompt_with_comma_separated_labels():
    prompt = Gemma4_Ollama._build_prompt("car, truck, bicycle")
    assert "car, truck, bicycle" in prompt


def test_build_prompt_strips_whitespace():
    prompt = Gemma4_Ollama._build_prompt("  helmet .   person  ")
    assert "helmet, person" in prompt


def test_build_prompt_empty_string_falls_through():
    prompt = Gemma4_Ollama._build_prompt("")
    assert "Detect every visible instance" in prompt


def test_normalize_label_strips_and_collapses_whitespace():
    assert Gemma4_Ollama._normalize_label("  Hard\tHat  ") == "Hard Hat"
    assert Gemma4_Ollama._normalize_label(None) == ""
    assert Gemma4_Ollama._normalize_label("person") == "person"


def test_normalize_label_handles_non_string():
    assert Gemma4_Ollama._normalize_label(42) == "42"
