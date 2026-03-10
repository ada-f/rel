from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from .rdkit_utils import canonical_smiles_from_smiles


# -----------------------------
# Parsing helpers
# -----------------------------

_SMILES_TAG_RE = re.compile(r"<smiles>\s*([^<\n\r]+)\s*(?:</smiles>)?", re.IGNORECASE)
_YESNO_TAG_RE = re.compile(r"<\s*(Yes|No)\s*>", re.IGNORECASE)


def extract_first_smiles_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _SMILES_TAG_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def extract_all_smiles_tags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [m.group(1).strip() for m in _SMILES_TAG_RE.finditer(text)]


def extract_yesno_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _YESNO_TAG_RE.search(text)
    if not m:
        return None
    return m.group(1).capitalize()


def extract_indices_and_motifs(text: str) -> Tuple[Optional[List[int]], Optional[Dict[str, str]]]:
    """
    Extract molecule indices and their corresponding motifs from REL-C5 response.

    Expected format:
    <indices>1, 3, 5</indices>
    <motif_1>CCCCCC</motif_1>
    <motif_3>C1CCCCC1</motif_3>
    <motif_5>c1ccccc1</motif_5>

    Returns:
        Tuple of (indices, motifs) where:
        - indices: list of ints, or None if not found
        - motifs: dict mapping str(index) -> SMILES, or None if not found
    """
    if not isinstance(text, str):
        return None, None

    # Extract indices
    indices_match = re.search(r"<indices>\s*([^<]+?)\s*</indices>", text, re.IGNORECASE)
    if not indices_match:
        return None, None

    # Parse comma-separated indices
    indices_str = indices_match.group(1)
    try:
        indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
    except ValueError:
        return None, None

    # Extract motifs for each index
    motifs = {}
    for idx in indices:
        motif_pattern = rf"<motif_{idx}>\s*([^<\n\r]+?)\s*</motif_{idx}>"
        motif_match = re.search(motif_pattern, text, re.IGNORECASE)
        if motif_match:
            motifs[str(idx)] = motif_match.group(1).strip()
        else:
            # Missing motif for this index
            return indices, None

    return indices, motifs


# -----------------------------
# OpenAI API caller (Chat Completions)
# -----------------------------

@dataclass(frozen=True)
class OpenAIChatConfig:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 120
    max_tokens: int = 5000
    temperature: float = 0.0
    system_prompt: str = (
        "You are a careful chemistry assistant. Follow the output format exactly. "
        "Do not include extra text."
    )
    # Basic retry/backoff
    max_retries: int = 6
    min_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    # Rate limit specific retry/backoff (for 429 errors)
    max_retries_rate_limit: int = 15
    rate_limit_min_backoff_s: float = 5.0
    rate_limit_max_backoff_s: float = 300.0


def call_openai_response(prompt: str, cfg: OpenAIChatConfig) -> str:
    """
    Uses the Responses API (/v1/responses). This is the recommended interface for GPT-5.x. :contentReference[oaicite:1]{index=1}
    """
    url = cfg.base_url.rstrip("/") + "/responses"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}

    payload = {
        "model": cfg.model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": cfg.system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        # If you want deterministic outputs for benchmarking:
        "temperature": cfg.temperature,
        # Limit output length:
        "max_output_tokens": cfg.max_tokens,
    }

    last_err = None
    for attempt in range(cfg.max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:2000]}")
            data = r.json()

            # Easiest stable extraction: use output_text if present (SDK exposes this),
            # otherwise concatenate output content chunks.
            if "output_text" in data and isinstance(data["output_text"], str):
                return data["output_text"].strip()

            chunks = []
            for item in data.get("output", []):
                for c in item.get("content", []):
                    if c.get("type") in ("output_text", "text") and "text" in c:
                        chunks.append(c["text"])
            return "".join(chunks).strip()

        except Exception as e:
            last_err = e
            backoff = min(cfg.max_backoff_s, cfg.min_backoff_s * (2 ** attempt))
            backoff *= (0.7 + 0.6 * random.random())
            time.sleep(backoff)

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


def call_openai_chat(prompt: str, cfg: OpenAIChatConfig) -> str:
    # works for GPT-4o-mini and Azure OpenAI
    url = cfg.base_url.rstrip("/") + "/chat/completions"

    # Azure uses api-key header, standard OpenAI uses Authorization Bearer
    is_azure = "azure" in cfg.base_url.lower() or "hms.edu" in cfg.base_url.lower()
    if is_azure:
        headers = {"api-key": cfg.api_key, "Content-Type": "application/json"}
        # Azure requires api-version query parameter
        url += "?api-version=2024-12-01-preview"
    else:
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}

    # GPT-5.x uses max_completion_tokens instead of max_tokens
    if cfg.model.startswith("gpt-5") or cfg.model.startswith("o1") or cfg.model.startswith("o3"):
        max_tokens_key = "max_completion_tokens"
    else:
        max_tokens_key = "max_tokens"

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.temperature,
        max_tokens_key: cfg.max_tokens,
    }

    last_err = None
    is_rate_limit = False
    max_attempts = cfg.max_retries

    for attempt in range(max_attempts):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)

            # Handle rate limiting (429) with special backoff
            if r.status_code == 429:
                is_rate_limit = True
                # Parse Retry-After header if available
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                        print(f"[Rate Limit] API requested retry after {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                    except ValueError:
                        wait_time = None
                else:
                    wait_time = None

                # If this is our first rate limit hit, extend max attempts
                if attempt < cfg.max_retries:
                    max_attempts = cfg.max_retries_rate_limit
                    print(f"[Rate Limit] Extending max attempts to {max_attempts} for rate limit handling")

                # Use Retry-After header if available, otherwise use exponential backoff
                if wait_time:
                    backoff = min(cfg.rate_limit_max_backoff_s, wait_time)
                else:
                    backoff = min(cfg.rate_limit_max_backoff_s, cfg.rate_limit_min_backoff_s * (2 ** attempt))
                    backoff = backoff * (0.7 + 0.6 * random.random())

                print(f"[Rate Limit] Waiting {backoff:.1f}s before retry...")
                time.sleep(backoff)
                continue

            # Handle other retryable errors
            if r.status_code in (500, 502, 503, 504):
                raise RuntimeError(f"retryable_status={r.status_code} body={r.text[:300]}")

            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            # For 400 errors, show the actual API error message
            if r.status_code == 400:
                try:
                    error_data = r.json()
                    error_msg = error_data.get("error", {}).get("message", r.text[:500])
                    raise RuntimeError(f"OpenAI API 400 Bad Request: {error_msg}")
                except:
                    raise RuntimeError(f"OpenAI API 400 Bad Request: {r.text[:500]}")
            # Don't retry on 400 errors (bad request) or 401 (unauthorized)
            if r.status_code in (400, 401):
                raise
            last_err = e
            # exponential backoff with jitter (for non-429 errors)
            backoff = min(cfg.max_backoff_s, cfg.min_backoff_s * (2 ** attempt))
            backoff = backoff * (0.7 + 0.6 * random.random())
            time.sleep(backoff)
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            backoff = min(cfg.max_backoff_s, cfg.min_backoff_s * (2 ** attempt))
            backoff = backoff * (0.7 + 0.6 * random.random())
            time.sleep(backoff)

    raise RuntimeError(f"OpenAI call failed after {max_attempts} retries: {last_err}")


# -----------------------------
# Claude API caller (Anthropic)
# -----------------------------

@dataclass(frozen=True)
class ClaudeChatConfig:
    model: str
    api_key: str
    base_url: str = "https://api.anthropic.com/v1"
    timeout_s: int = 120
    max_tokens: int = 5000
    temperature: float = 0.0
    system_prompt: str = (
        "You are a careful chemistry assistant. Follow the output format exactly. "
        "Do not include extra text."
    )
    # Basic retry/backoff
    max_retries: int = 6
    min_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    # Rate limit specific retry/backoff (for 429 errors)
    max_retries_rate_limit: int = 15
    rate_limit_min_backoff_s: float = 5.0
    rate_limit_max_backoff_s: float = 300.0


def call_claude(prompt: str, cfg: ClaudeChatConfig) -> str:
    """
    Calls the Anthropic Claude API using the messages endpoint.
    """
    url = cfg.base_url.rstrip("/") + "/messages"
    headers = {
        "x-api-key": cfg.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    payload = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "system": cfg.system_prompt,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    last_err = None
    is_rate_limit = False
    max_attempts = cfg.max_retries

    for attempt in range(max_attempts):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_s)

            # Handle rate limiting (429) with special backoff
            if r.status_code == 429:
                is_rate_limit = True
                # Parse Retry-After header if available
                retry_after = r.headers.get("Retry-After") or r.headers.get("retry-after")
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                        print(f"[Rate Limit] API requested retry after {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                    except ValueError:
                        wait_time = None
                else:
                    wait_time = None

                # If this is our first rate limit hit, extend max attempts
                if attempt < cfg.max_retries:
                    max_attempts = cfg.max_retries_rate_limit
                    print(f"[Rate Limit] Extending max attempts to {max_attempts} for rate limit handling")

                # Use Retry-After header if available, otherwise use exponential backoff
                if wait_time:
                    backoff = min(cfg.rate_limit_max_backoff_s, wait_time)
                else:
                    backoff = min(cfg.rate_limit_max_backoff_s, cfg.rate_limit_min_backoff_s * (2 ** attempt))
                    backoff = backoff * (0.7 + 0.6 * random.random())

                print(f"[Rate Limit] Waiting {backoff:.1f}s before retry...")
                time.sleep(backoff)
                continue

            # Handle other retryable errors
            if r.status_code in (500, 502, 503, 504):
                raise RuntimeError(f"retryable_status={r.status_code} body={r.text[:300]}")

            r.raise_for_status()
            data = r.json()

            # Extract text from content array
            content = data.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                return "".join(text_parts).strip()
            elif isinstance(content, str):
                return content.strip()
            else:
                raise RuntimeError(f"Unexpected Claude response format: {data}")

        except requests.exceptions.HTTPError as e:
            # For 400/404 errors, show the actual API error message
            if r.status_code in (400, 404):
                try:
                    error_data = r.json()
                    error_msg = error_data.get("error", {}).get("message", r.text[:500])
                    raise RuntimeError(f"Claude API {r.status_code} Error: {error_msg}. Full response: {r.text[:1000]}")
                except:
                    raise RuntimeError(f"Claude API {r.status_code} Error: {r.text[:1000]}")
            # Don't retry on 400 errors (bad request) or 401 (unauthorized) or 404 (not found)
            if r.status_code in (400, 401, 404):
                raise
            last_err = e
            # exponential backoff with jitter (for non-429 errors)
            backoff = min(cfg.max_backoff_s, cfg.min_backoff_s * (2 ** attempt))
            backoff = backoff * (0.7 + 0.6 * random.random())
            time.sleep(backoff)
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            backoff = min(cfg.max_backoff_s, cfg.min_backoff_s * (2 ** attempt))
            backoff = backoff * (0.7 + 0.6 * random.random())
            time.sleep(backoff)

    raise RuntimeError(f"Claude call failed after {max_attempts} retries: {last_err}")


# -----------------------------
# Gemini API caller (Google)
# -----------------------------

@dataclass(frozen=True)
class GeminiChatConfig:
    model: str
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    timeout_s: int = 600
    # For thinking models (like gemini-3-pro-preview), we need high token limits
    # IMPORTANT: maxOutputTokens controls TOTAL generation (thinking + answer combined)
    # For Gemini 2.5: Use thinkingBudget (soft limit on thinking tokens)
    # For Gemini 3.x: Use thinkingLevel (controls reasoning depth: "low", "medium", "high")
    max_tokens: int = 8000  # Total budget for thinking + answer
    thinking_budget: int = 4000  # For Gemini 2.5: Portion that can be used for thinking
    thinking_level: Optional[str] = None  # For Gemini 3.x: "low", "medium", or "high"
    disable_thinking: bool = False  # If True, don't include thinkingConfig in the request
    temperature: float = 0.0
    system_prompt: str = (
        "You are a careful chemistry assistant. Follow the output format exactly. "
        "Do not include extra text."
    )
    # Basic retry/backoff
    max_retries: int = 6
    min_backoff_s: float = 1.0
    max_backoff_s: float = 30.0
    # Rate limit specific retry/backoff (for 429 errors)
    max_retries_rate_limit: int = 15
    rate_limit_min_backoff_s: float = 5.0
    rate_limit_max_backoff_s: float = 300.0


def call_gemini(prompt: str, cfg: GeminiChatConfig) -> str:
    """
    Calls the Google Gemini API using the generateContent endpoint.
    Uses v1beta which supports system_instruction (v1 does not support it).
    """
    url = f"{cfg.base_url.rstrip('/')}/models/{cfg.model}:generateContent"
    params = {"key": cfg.api_key}
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": cfg.temperature,
            "maxOutputTokens": cfg.max_tokens,
        }
    }

    # Add thinkingConfig if not disabled
    # For Gemini 3.x models: Use thinkingLevel to control reasoning depth
    # For Gemini 2.5 models: Use thinkingBudget (soft limit)
    # We do NOT set includeThoughts=True because we only want the final answer,
    # not the thinking process. The model will still think internally.
    if not cfg.disable_thinking:
        thinking_config = {}

        # Gemini 3.x: Use thinkingLevel if specified (preferred for token control)
        if cfg.thinking_level:
            thinking_config["thinkingLevel"] = cfg.thinking_level
        else:
            # Gemini 2.5: Use thinkingBudget (soft limit - model can exceed it)
            thinking_config["thinkingBudget"] = cfg.thinking_budget

        payload["generationConfig"]["thinkingConfig"] = thinking_config

    # Add system instruction if provided (v1beta supports system_instruction)
    if cfg.system_prompt:
        payload["system_instruction"] = {
            "parts": [{
                "text": cfg.system_prompt
            }]
        }
    
    last_err = None
    is_rate_limit = False
    max_attempts = cfg.max_retries

    for attempt in range(max_attempts):
        try:
            r = requests.post(url, params=params, json=payload, timeout=cfg.timeout_s)

            # Handle rate limiting (429) with special backoff
            if r.status_code == 429:
                is_rate_limit = True
                # Parse Retry-After header if available
                retry_after = r.headers.get("Retry-After") or r.headers.get("retry-after")
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                        print(f"[Rate Limit] API requested retry after {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                    except ValueError:
                        wait_time = None
                else:
                    wait_time = None

                # If this is our first rate limit hit, extend max attempts
                if attempt < cfg.max_retries:
                    max_attempts = cfg.max_retries_rate_limit
                    print(f"[Rate Limit] Extending max attempts to {max_attempts} for rate limit handling")

                # Use Retry-After header if available, otherwise use exponential backoff
                if wait_time:
                    backoff = min(cfg.rate_limit_max_backoff_s, wait_time)
                else:
                    backoff = min(cfg.rate_limit_max_backoff_s, cfg.rate_limit_min_backoff_s * (2 ** attempt))
                    backoff = backoff * (0.7 + 0.6 * random.random())

                print(f"[Rate Limit] Waiting {backoff:.1f}s before retry...")
                time.sleep(backoff)
                continue

            # Handle other retryable errors
            if r.status_code in (500, 502, 503, 504):
                raise RuntimeError(f"retryable_status={r.status_code} body={r.text[:300]}")

            r.raise_for_status()
            data = r.json()

            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"No candidates in Gemini response: {data}")

            first = candidates[0]
            finish_reason = first.get("finishReason")

            content = first.get("content") or {}
            parts = content.get("parts") or []

            text_parts = []
            thinking_parts = []
            all_text_parts = []  # Collect all text regardless of thought field
            
            # Debug: log what we're getting (only on first attempt to avoid spam)
            import json
            if attempt == 0 and not parts:
                print(f"[DEBUG] No parts in response. Finish reason: {finish_reason}")
                print(f"[DEBUG] Full candidate keys: {list(first.keys()) if isinstance(first, dict) else type(first)}")
                print(f"[DEBUG] Content structure: {json.dumps(content, indent=2)[:2000] if isinstance(content, dict) else str(content)[:500]}")
                # Check usage metadata to see token usage
                usage = data.get("usageMetadata", {})
                if usage:
                    print(f"[DEBUG] Token usage: {json.dumps(usage, indent=2)}")
                    thoughts_tokens = usage.get("thoughtsTokenCount", 0)
                    candidates_tokens = usage.get("candidatesTokenCount", 0)
                    total_tokens = usage.get("totalTokenCount", 0)
                    print(f"[DEBUG] Thoughts: {thoughts_tokens}, Candidates: {candidates_tokens}, Total: {total_tokens}")
                    # Check if thinking consumed all budget
                    if thoughts_tokens > 0 and candidates_tokens == 0:
                        print(f"[WARNING] Thinking used {thoughts_tokens} tokens but no output generated. This suggests thinking consumed the budget.")
                        print(f"[WARNING] Current config: thinkingBudget={cfg.thinking_budget}, maxOutputTokens={cfg.max_tokens}")
                # Log the full first candidate for deeper inspection
                print(f"[DEBUG] Full first candidate (first 5000 chars): {json.dumps(first, indent=2)[:5000]}")
                # Also check if there are any other fields we're missing
                if "groundingMetadata" in first:
                    print(f"[DEBUG] Found groundingMetadata in candidate")
                if "safetyRatings" in first:
                    print(f"[DEBUG] Found safetyRatings in candidate")
            
            for part in parts:
                if "text" in part:
                    text_content = part["text"]
                    
                    # For thinking models, parts can have a "thought" field
                    # If thought=True, it's thinking content; if thought=False or missing, it's the answer
                    is_thinking = part.get("thought", False)
                    
                    if is_thinking:
                        # Store thinking parts separately for debugging
                        thinking_parts.append(text_content)
                    else:
                        # This is the actual answer - ONLY use non-thinking parts
                        text_parts.append(text_content)
                else:
                    # Debug: log parts without text (only first time)
                    if attempt == 0:
                        print(f"[DEBUG] Part without text field: {list(part.keys()) if isinstance(part, dict) else type(part)}")

            # ONLY use non-thinking parts (the actual answers)
            text = "".join(text_parts).strip()
            
            # If no non-thinking parts but we have parts, check if thought field is working correctly
            if not text and parts and attempt == 0:
                # Check if all parts are marked as thinking (might be a bug)
                parts_with_text = [p for p in parts if "text" in p]
                if parts_with_text:
                    thinking_marked = [p for p in parts_with_text if p.get("thought", False)]
                    non_thinking_marked = [p for p in parts_with_text if not p.get("thought", False)]
                    
                    print(f"[DEBUG] Parts breakdown: {len(thinking_marked)} thinking, {len(non_thinking_marked)} non-thinking")
                    
                    # If ALL parts are marked as thinking, the thought field might not be working
                    # In this case, try to find the actual answer by looking for parts that don't
                    # start with thinking markers (like "**Finding", "**My Approach", etc.)
                    if len(thinking_marked) == len(parts_with_text) and len(thinking_marked) > 0:
                        print(f"[WARNING] All parts marked as thinking - thought field may not be working correctly")
                        # Try to find parts that look like answers (not thinking)
                        # Thinking often starts with headers like "**Finding", "**My Approach", "**Analyzing"
                        answer_like_parts = []
                        for part in parts_with_text:
                            part_text = part.get("text", "")
                            # Skip parts that start with thinking markers
                            if not (part_text.strip().startswith("**") or 
                                   part_text.strip().startswith("Okay, so") or
                                   part_text.strip().startswith("My Approach") or
                                   part_text.strip().startswith("Analyzing") or
                                   "I've been tasked" in part_text[:100] or
                                   "Let's break this down" in part_text[:100]):
                                answer_like_parts.append(part_text)
                        
                        if answer_like_parts:
                            text = "".join(answer_like_parts).strip()
                            print(f"[INFO] Extracted answer from {len(answer_like_parts)} answer-like parts (thought field not working)")
                        else:
                            # If we can't find answer-like parts, the model might only be generating thinking
                            # In this case, we have no choice but to return empty or use the last part
                            print(f"[WARNING] Could not identify answer parts - model may only be generating thinking content")
                    elif len(non_thinking_marked) > 0:
                        # We have non-thinking parts but text is empty - this shouldn't happen
                        print(f"[ERROR] Found {len(non_thinking_marked)} non-thinking parts but text extraction failed")
                        # Force extract from non-thinking parts
                        text = "".join([p.get("text", "") for p in non_thinking_marked]).strip()

            # If we have no text but we have parts, check if they're all marked as thinking
            # This might indicate the thought field isn't working correctly
            if not text and parts and attempt == 0:
                # Check if all parts with text are marked as thinking
                parts_with_text = [p for p in parts if "text" in p]
                thinking_count = sum(1 for p in parts_with_text if p.get("thought", False))
                non_thinking_count = len(parts_with_text) - thinking_count
                print(f"[DEBUG] Parts analysis: {len(parts_with_text)} parts with text, {thinking_count} marked as thinking, {non_thinking_count} marked as answer")
                if thinking_count > 0 and non_thinking_count == 0:
                    print(f"[WARNING] All {thinking_count} text parts are marked as thinking. The model may not be generating answer parts.")
                    # Show a sample of the thinking to see if answer is embedded
                    if thinking_parts:
                        sample = thinking_parts[0][:500] if len(thinking_parts[0]) > 500 else thinking_parts[0]
                        print(f"[DEBUG] Sample thinking content (first 500 chars): {sample}")
            
            # If we hit MAX_TOKENS and Gemini did not return any parts/text,
            # this might indicate the thinking consumed all tokens
            # But sometimes content might still exist in a different structure
            if not text and finish_reason == "MAX_TOKENS":
                # If we have thinking parts, try to use them (as last resort)
                if thinking_parts:
                    # Use the last thinking part as it might contain the answer
                    text = thinking_parts[-1].strip()
                    if text:
                        print(f"[INFO] MAX_TOKENS hit but extracted text from last thinking part ({len(text)} chars)")
                        return text
                
                # Check if content has any text directly (different structure)
                if isinstance(content, dict):
                    # Sometimes text might be in content directly
                    for key in ["text", "response", "output"]:
                        if key in content and isinstance(content[key], str) and content[key].strip():
                            text = content[key].strip()
                            print(f"[INFO] MAX_TOKENS hit but found text in content.{key} ({len(text)} chars)")
                            return text
                
                # If still no text, try to extract thinking content for debugging
                import json
                usage = data.get("usageMetadata", {})
                thoughts_tokens = usage.get("thoughtsTokenCount", 0) if usage else 0
                
                print(f"[WARNING] MAX_TOKENS reached with no extractable text. Parts: {len(parts)}, Thinking parts: {len(thinking_parts)}")
                
                # If thinking tokens were used, try to find thinking content in the response
                if thoughts_tokens > 0:
                    print(f"[DEBUG] Thinking tokens used: {thoughts_tokens}, attempting to extract thinking content...")
                    
                    # Check if thinking is in a different location in the response
                    thinking_content = None
                    
                    # Check candidate for thinking fields
                    if "thinking" in first:
                        thinking_content = first.get("thinking")
                        print(f"[DEBUG] Found 'thinking' field in candidate")
                    
                    # Check content for thinking
                    if not thinking_content and isinstance(content, dict):
                        if "thinking" in content:
                            thinking_content = content.get("thinking")
                            print(f"[DEBUG] Found 'thinking' field in content")
                        # Check all keys in content for thinking-related fields
                        for key in content.keys():
                            if "think" in key.lower():
                                thinking_content = content.get(key)
                                print(f"[DEBUG] Found thinking-related field '{key}' in content")
                                break
                    
                    # Check the full data structure for thinking
                    if not thinking_content:
                        # Recursively search for thinking content
                        def find_thinking(obj, path=""):
                            if isinstance(obj, dict):
                                for key, value in obj.items():
                                    if "think" in key.lower() and isinstance(value, str):
                                        return value
                                    if isinstance(value, (dict, list)):
                                        result = find_thinking(value, f"{path}.{key}")
                                        if result:
                                            return result
                            elif isinstance(obj, list):
                                for i, item in enumerate(obj):
                                    result = find_thinking(item, f"{path}[{i}]")
                                    if result:
                                        return result
                            return None
                        
                        thinking_content = find_thinking(data)
                        if thinking_content:
                            print(f"[DEBUG] Found thinking content in nested structure")
                    
                    # Print thinking content if found
                    if thinking_content:
                        if isinstance(thinking_content, str):
                            print(f"[THINKING DEBUG] Thinking content ({len(thinking_content)} chars):")
                            print(thinking_content[:5000])  # Print first 5000 chars
                        else:
                            print(f"[THINKING DEBUG] Thinking content (type: {type(thinking_content)}):")
                            print(json.dumps(thinking_content, indent=2)[:5000])
                    else:
                        print(f"[DEBUG] Could not find thinking content in response structure")
                        print(f"[DEBUG] Full response structure (first 10000 chars): {json.dumps(data, indent=2)[:10000]}")
                
                return ""

            # For thinking models or when no text is found, provide detailed error
            if not text:
                # Log detailed structure for debugging
                import json
                
                # Also try to extract thinking content for debugging
                usage = data.get("usageMetadata", {})
                thoughts_tokens = usage.get("thoughtsTokenCount", 0) if usage else 0
                
                if thoughts_tokens > 0 and not thinking_parts:
                    print(f"[DEBUG] Thinking tokens used ({thoughts_tokens}) but no thinking parts found. Searching response...")
                    # Try to find thinking in the response structure (same logic as MAX_TOKENS case)
                    def find_thinking(obj):
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if "think" in key.lower() and isinstance(value, str):
                                    return value
                                if isinstance(value, (dict, list)):
                                    result = find_thinking(value)
                                    if result:
                                        return result
                        elif isinstance(obj, list):
                            for item in obj:
                                result = find_thinking(item)
                                if result:
                                    return result
                        return None
                    
                    thinking_content = find_thinking(data)
                    if thinking_content:
                        print(f"[THINKING DEBUG] Found thinking content ({len(thinking_content)} chars):")
                        print(thinking_content[:5000])
                    elif thinking_parts:
                        print(f"[THINKING DEBUG] Thinking parts found ({len(thinking_parts)} parts):")
                        for i, tp in enumerate(thinking_parts[:3]):  # Show first 3 thinking parts
                            print(f"[THINKING DEBUG] Part {i+1} ({len(tp)} chars): {tp[:1000]}")
                
                # Log detailed structure for debugging
                debug_info = {
                    "finish_reason": finish_reason,
                    "parts_count": len(parts),
                    "parts_structure": [list(p.keys()) if isinstance(p, dict) else type(p).__name__ for p in parts] if parts else "empty",
                    "content_keys": list(content.keys()) if isinstance(content, dict) else type(content).__name__,
                    "candidate_keys": list(first.keys()) if isinstance(first, dict) else type(first).__name__,
                    "thinking_parts_found": len(thinking_parts) if 'thinking_parts' in locals() else 0,
                }
                # Try alternative extraction: maybe content is a string or has different structure
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, dict):
                    # Check if text is directly in content
                    if "text" in content:
                        text = str(content["text"]).strip()
                    # Check all string values in content
                    for key, value in content.items():
                        if isinstance(value, str) and value.strip() and key != "role":
                            text = value.strip()
                            break
                
                if not text:
                    # For debugging: print the actual response structure
                    error_msg = f"No text answer in Gemini response.\n"
                    error_msg += f"Debug info: {json.dumps(debug_info, indent=2)}\n"
                    if 'thinking_parts' in locals() and thinking_parts:
                        error_msg += f"Found {len(thinking_parts)} thinking parts but no answer parts.\n"
                        error_msg += f"First thinking part (first 500 chars): {thinking_parts[0][:500]}\n"
                    error_msg += f"Parts structure: {json.dumps(parts[:3], indent=2) if parts else '[]'}\n"  # First 3 parts
                    error_msg += f"Full response (first 3000 chars): {json.dumps(data, indent=2)[:3000]}"
                    raise RuntimeError(error_msg)

            return text
            
        except requests.exceptions.HTTPError as e:
            # For 400 errors, show the actual API error message
            if r.status_code == 400:
                try:
                    error_data = r.json()
                    error_msg = error_data.get("error", {}).get("message", r.text[:500])
                    raise RuntimeError(f"Gemini API 400 Bad Request: {error_msg}")
                except:
                    raise RuntimeError(f"Gemini API 400 Bad Request: {r.text[:500]}")
            # Don't retry on 400 errors (bad request) or 401 (unauthorized)
            if r.status_code in (400, 401):
                raise
            last_err = e
            # exponential backoff with jitter (for non-429 errors)
            backoff = min(cfg.max_backoff_s, cfg.min_backoff_s * (2 ** attempt))
            backoff = backoff * (0.7 + 0.6 * random.random())
            time.sleep(backoff)
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            backoff = min(cfg.max_backoff_s, cfg.min_backoff_s * (2 ** attempt))
            backoff = backoff * (0.7 + 0.6 * random.random())
            time.sleep(backoff)

    raise RuntimeError(f"Gemini call failed after {max_attempts} retries: {last_err}")


# -----------------------------
# Scoring
# -----------------------------

def score_instance(instance: Dict[str, Any], model_text: str) -> Dict[str, Any]:
    task = instance["task"]
    gold = instance["answer"]

    # Map unified format task names to internal names
    task_mapping = {
        "REL-C1": "q2_isomer_set_yes_no",
        "REL-C2": "q1a_largest_common_motif_chembl",  # Both Q1a and Q1b map to same scoring
        "REL-C3": "q3_missing_isomers",
        "REL-C4": "q4_avoid_scaffolds",
        "REL-C5": "q5_constraint_satisfaction_selection",
    }
    task = task_mapping.get(task, task)

    # Q1a and Q1b both use the same scoring logic as Q1
    if task in ("q1_largest_common_motif", "q1a_largest_common_motif_chembl", "q1b_largest_common_motif_scaffold"):
        pred = extract_first_smiles_tag(model_text)
        pred_can = canonical_smiles_from_smiles(pred, isomeric=False) if pred else None
        gold_can = canonical_smiles_from_smiles(gold["smiles"], isomeric=False)
        correct = (pred_can is not None) and (gold_can is not None) and (pred_can == gold_can)
        return {"pred": pred, "pred_canonical": pred_can, "gold": gold["smiles"], "gold_canonical": gold_can, "correct": correct}

    if task == "q2_isomer_set_yes_no":
        pred = extract_yesno_tag(model_text)
        gold_label = gold["label"]
        correct = (pred is not None) and (pred == gold_label)
        return {"pred": pred, "gold": gold_label, "correct": correct}

    if task == "q3_missing_isomers":
        pred_list = extract_all_smiles_tags(model_text)
        pred_can = sorted({canonical_smiles_from_smiles(s, isomeric=False) for s in pred_list if canonical_smiles_from_smiles(s, isomeric=False)})
        gold_can = sorted({canonical_smiles_from_smiles(s, isomeric=False) for s in gold["missing_smiles"] if canonical_smiles_from_smiles(s, isomeric=False)})

        pred_set = set(pred_can)
        gold_set = set(gold_can)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        exact = (pred_set == gold_set)

        return {
            "pred": pred_list,
            "pred_canonical": pred_can,
            "gold_canonical": gold_can,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "correct": exact,
        }

    if task == "q4_avoid_scaffolds":
        # Use enhanced evaluation from evaluation.py
        from .evaluation import evaluate_c4_response
        metadata = instance.get("metadata", {})
        # Pass empty string for question (not used in evaluation)
        return evaluate_c4_response("", gold, model_text, metadata)

    if task == "q5_constraint_satisfaction_selection":
        from rdkit import Chem
        from .functional_groups import count_functional_groups, count_aromatic_rings

        # Extract metadata
        metadata = instance.get("metadata", {})
        constraint_type = metadata.get("constraint_type")
        target_value = metadata.get("target_value")
        k_molecules = metadata.get("k_molecules")
        min_motif_atoms = metadata.get("min_motif_atoms", 6)
        molecules = gold.get("molecules", [])

        # For reference (not used for validation, but kept for comparison)
        gold_indices = gold.get("selected_molecule_indices", [])
        gold_motifs = gold.get("selected_motifs", {})

        # Parse LLM response
        pred_indices, pred_motifs = extract_indices_and_motifs(model_text)

        # Validation checks
        errors = []
        constraint_satisfied = False
        actual_total = None

        # 1. Check if we extracted indices and motifs
        if pred_indices is None or pred_motifs is None:
            errors.append("Failed to parse response")
            return {
                "pred_indices": pred_indices,
                "pred_motifs": pred_motifs,
                "gold_indices": gold_indices,
                "gold_motifs": gold_motifs,
                "correct": False,
                "constraint_satisfied": False,
                "errors": errors,
            }

        # 2. Check selected exactly k molecules
        if len(pred_indices) != k_molecules:
            errors.append(f"Selected {len(pred_indices)} molecules, expected {k_molecules}")

        # 3. Check each selected molecule has a motif
        if set(pred_motifs.keys()) != set(str(i) for i in pred_indices):
            errors.append("Missing motifs for some selected molecules")

        # 4. Validate each motif is a substructure and has enough atoms
        from rdkit import Chem
        for idx_str, motif_smiles in pred_motifs.items():
            try:
                mol_idx = int(idx_str)
                if mol_idx < 0 or mol_idx >= len(molecules):
                    errors.append(f"Invalid molecule index: {mol_idx}")
                    continue

                mol_smiles = molecules[mol_idx]
                motif_mol = Chem.MolFromSmiles(motif_smiles)
                parent_mol = Chem.MolFromSmiles(mol_smiles)

                if motif_mol is None:
                    errors.append(f"Invalid SMILES for motif at index {mol_idx}: {motif_smiles}")
                    continue

                if parent_mol is None:
                    errors.append(f"Invalid SMILES for parent molecule at index {mol_idx}")
                    continue

                # Check substructure
                if not parent_mol.HasSubstructMatch(motif_mol):
                    errors.append(f"Motif at index {mol_idx} is not a substructure of the parent molecule")

                # Check minimum atoms
                num_atoms = motif_mol.GetNumHeavyAtoms()
                if num_atoms < min_motif_atoms:
                    errors.append(f"Motif at index {mol_idx} has {num_atoms} atoms, minimum is {min_motif_atoms}")

            except Exception as e:
                errors.append(f"Error validating motif at index {idx_str}: {e}")

        # 5. Check constraint satisfaction (only if no validation errors so far)
        if not errors:
            try:
                actual_total = 0
                for motif_smiles in pred_motifs.values():
                    motif_mol = Chem.MolFromSmiles(motif_smiles)
                    if motif_mol is None:
                        continue

                    if constraint_type == "total_aromatic_rings":
                        value = count_aromatic_rings(motif_mol)
                    elif constraint_type and constraint_type.startswith("total_"):
                        # Extract functional group name
                        fg_key = constraint_type[6:]  # Remove "total_" prefix
                        if fg_key.endswith("s"):
                            fg_key = fg_key[:-1]  # Remove plural "s"
                        fg_counts = count_functional_groups(motif_mol)
                        value = fg_counts.get(fg_key, 0)
                    else:
                        fg_counts = count_functional_groups(motif_mol)
                        value = fg_counts.get(constraint_type, 0)

                    actual_total += value

                constraint_satisfied = (actual_total == target_value)
                if not constraint_satisfied:
                    errors.append(f"Constraint not satisfied: {constraint_type}={actual_total}, expected {target_value}")

            except Exception as e:
                errors.append(f"Error checking constraint: {e}")

        # Overall correct if constraint is satisfied and no errors
        correct = constraint_satisfied and len(errors) == 0

        return {
            "pred_indices": pred_indices,
            "pred_motifs": pred_motifs,
            "gold_indices": gold_indices,  # Reference only
            "gold_motifs": gold_motifs,    # Reference only
            "correct": correct,
            "constraint_satisfied": constraint_satisfied,
            "actual_total": actual_total,
            "expected_total": target_value,
            "errors": errors if errors else None,
        }

    return {"error": f"Unknown task: {task}", "correct": False}


def summarize_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Overall accuracy and by-task
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    summary: Dict[str, Any] = {}
    total = len(rows)
    summary["total"] = total
    summary["overall_accuracy"] = sum(1 for r in rows if r.get("score", {}).get("correct")) / total if total else 0.0

    task_summaries = {}
    for t, trs in by_task.items():
        acc = sum(1 for r in trs if r.get("score", {}).get("correct")) / len(trs) if trs else 0.0
        task_summaries[t] = {"n": len(trs), "accuracy": acc}
    summary["by_task_accuracy"] = task_summaries
    return summary


# -----------------------------
# Runner
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="data/dataset.jsonl")
    ap.add_argument("--out", type=str, default="runs/run.jsonl")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--provider", type=str, choices=["openai", "claude", "gemini"], default="openai",
                    help="API provider to use: openai, claude, or gemini")
    ap.add_argument("--api_key_env", type=str, default=None,
                    help="Environment variable name for API key (default: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY based on provider)")
    ap.add_argument("--base_url", type=str, default=None,
                    help="Base URL for API (default: provider-specific default)")
    ap.add_argument("--max_tokens", type=int, default=5000)
    ap.add_argument("--thinking_budget", type=int, default=5000,
                    help="For Gemini 2.5: soft limit on thinking tokens")
    ap.add_argument("--thinking_level", type=str, choices=["low", "medium", "high"], default=None,
                    help="For Gemini 3.x: control reasoning depth (low/medium/high). Recommended for token control.")
    ap.add_argument("--disable_thinking", action="store_true", help="Disable thinkingConfig for Gemini (useful if thinking consumes all tokens)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--test_mode", type=int, default=None, help="Test mode: sample N instances from each task type (default: None = process all)")
    ap.add_argument("--test_seed", type=int, default=42, help="Random seed for test mode sampling")
    ap.add_argument("--task_filter", type=str, nargs="+", default=None,
                    help="Process only specific task types (e.g., q1a_largest_common_motif_chembl q1b_largest_common_motif_scaffold q2_isomer_set_yes_no q3_missing_isomers)")
    args = ap.parse_args()

    # Set default API key environment variable based on provider
    if args.api_key_env is None:
        provider_key_map = {
            "openai": "OPENAI_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY"
        }
        args.api_key_env = provider_key_map[args.provider]

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {args.api_key_env}")

    # Set default base URL based on provider if not specified
    if args.base_url is None:
        provider_url_map = {
            "openai": "https://api.openai.com/v1",
            "claude": "https://api.anthropic.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta"  # v1beta supports system_instruction
        }
        args.base_url = provider_url_map[args.provider]

    # Create appropriate config based on provider
    if args.provider == "openai":
        cfg = OpenAIChatConfig(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        # Use chat completions for Azure, responses API for standard OpenAI
        if "azure" in args.base_url.lower() or "hms.edu" in args.base_url.lower():
            call_func = call_openai_chat
        else:
            call_func = call_openai_response
    elif args.provider == "claude":
        cfg = ClaudeChatConfig(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        call_func = call_claude
    elif args.provider == "gemini":
        cfg = GeminiChatConfig(
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            thinking_budget=args.thinking_budget,
            thinking_level=args.thinking_level,
            disable_thinking=args.disable_thinking,
            temperature=args.temperature,
        )
        call_func = call_gemini
    else:
        raise RuntimeError(f"Unknown provider: {args.provider}")

    ds_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all instances
    all_instances = []
    with ds_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            all_instances.append(json.loads(line))

    # Filter by task type if requested
    if args.task_filter:
        task_filter_set = set(args.task_filter)
        all_instances = [inst for inst in all_instances if inst["task"] in task_filter_set]
        print(f"[TASK FILTER] Processing {len(all_instances)} instances matching tasks: {args.task_filter}")

    # Filter instances for test mode
    if args.test_mode is not None:
        # Group by task type
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for inst in all_instances:
            by_task.setdefault(inst["task"], []).append(inst)

        # Sample N instances from each task type
        rng = random.Random(args.test_seed)
        selected_instances = []
        for task, instances in by_task.items():
            n_sample = min(args.test_mode, len(instances))
            selected = rng.sample(instances, n_sample)
            selected_instances.extend(selected)
            print(f"[TEST MODE] Selected {len(selected)} instances from {task} (out of {len(instances)} total)")

        all_instances = selected_instances
        print(f"[TEST MODE] Processing {len(all_instances)} total instances (test_mode={args.test_mode})")

    # Load existing results for resumption
    processed_ids = set()
    rows = []
    if out_path.exists():
        print(f"[RESUME] Found existing results file: {out_path}")
        with out_path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    row = json.loads(line)
                    processed_ids.add(row["id"])
                    rows.append(row)  # Keep existing results in memory for summary
                except json.JSONDecodeError:
                    print(f"[RESUME WARNING] Skipping invalid JSON line in existing results")
                    continue
        print(f"[RESUME] Loaded {len(processed_ids)} existing results")

    # Filter to unprocessed instances
    original_count = len(all_instances)
    all_instances = [inst for inst in all_instances if inst["id"] not in processed_ids]
    skipped_count = original_count - len(all_instances)

    if skipped_count > 0:
        print(f"[RESUME] Skipping {skipped_count} already-processed instances")
    print(f"[PROGRESS] Processing {len(all_instances)} remaining instances")

    if len(all_instances) == 0:
        print(f"[RESUME] All instances already processed, nothing to do")
    else:
        # Open in append mode to preserve existing results
        with out_path.open("a", encoding="utf-8") as f_out:
            for idx, inst in enumerate(all_instances, 1):
                # Handle both "prompt" (old format) and "question" (unified format)
                prompt_text = inst.get("question") or inst.get("prompt")
                if not prompt_text:
                    raise ValueError(f"Instance {inst['id']} missing both 'question' and 'prompt' fields")

                print(f"[PROGRESS] Processing instance {idx}/{len(all_instances)} (ID: {inst['id']}, Task: {inst['task']})")

                text = call_func(prompt_text, cfg)
                score = score_instance(inst, text)

                # Get n_molecules from various possible locations
                n_mols = inst.get("n_molecules")
                if n_mols is None and "metadata" in inst:
                    n_mols = inst["metadata"].get("n_molecules")
                if n_mols is None and "answer" in inst and "molecules" in inst["answer"]:
                    n_mols = len(inst["answer"]["molecules"])

                row = {
                    "id": inst["id"],
                    "task": inst["task"],
                    "n_molecules": n_mols,
                    "model": args.model,
                    "response_text": text,
                    "score": score,
                }
                rows.append(row)
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()  # Ensure data is written immediately

                print(f"[PROGRESS] Completed instance {idx}/{len(all_instances)} - Correct: {score.get('correct', False)}")

    summary = summarize_scores(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
