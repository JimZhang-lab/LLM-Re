"""Utility functions for RE module - lightweight version extracted from utils.py"""
from __future__ import annotations
import asyncio
import html
import json
import logging
import os
import re
import time
from hashlib import md5
from typing import Any, List, Optional
import numpy as np

# Initialize logger
logger = logging.getLogger("lightrag.re")
logger.propagate = False
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def compute_args_hash(*args: Any) -> str:
    """Compute a hash for the given arguments with safe Unicode handling."""
    args_str = "".join([str(arg) for arg in args])
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Compute a unique ID for a given content string."""
    return prefix + compute_args_hash(content)


def generate_cache_key(mode: str, cache_type: str, hash_value: str) -> str:
    """Generate a flattened cache key."""
    return f"{mode}:{cache_type}:{hash_value}"


def split_string_by_multi_markers(content: str, markers: List[str]) -> List[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """Sanitize text to ensure safe UTF-8 encoding."""
    if not text:
        return text
    
    try:
        text = text.strip()
        if not text:
            return text
        
        # Test encoding
        text.encode("utf-8")
        
        # Remove surrogate characters
        sanitized = ""
        for char in text:
            code_point = ord(char)
            if 0xD800 <= code_point <= 0xDFFF:
                sanitized += replacement_char
                continue
            elif code_point == 0xFFFE or code_point == 0xFFFF:
                sanitized += replacement_char
                continue
            else:
                sanitized += char
        
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", replacement_char, sanitized)
        
        # Test final encoding
        sanitized.encode("utf-8")
        
        # Unescape HTML
        sanitized = html.unescape(sanitized)
        
        # Remove control characters
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", sanitized)
        
        return sanitized.strip()
        
    except UnicodeEncodeError as e:
        error_msg = f"Text contains uncleanable UTF-8 encoding issues: {str(e)[:100]}"
        logger.error(f"Text sanitization failed: {error_msg}")
        raise ValueError(error_msg) from e
    except Exception as e:
        logger.error(f"Unexpected error in text sanitization: {e}")
        return text


def normalize_extracted_info(name: str, remove_inner_quotes: bool = False) -> str:
    """Normalize extracted entity/relationship names."""
    # Clean HTML tags
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)
    
    # Full-width to half-width conversion
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    
    # Symbols
    name = name.replace("－", "-").replace("＋", "+")
    name = name.replace("（", "(").replace("）", ")")
    name = name.replace("［", "[").replace("］", "]")
    name = name.replace("｛", "{").replace("｝", "}")
    name = name.replace("，", ",").replace("。", ".")
    name = name.replace("；", ";").replace("：", ":")
    name = name.replace("？", "?").replace("！", "!")
    name = name.replace("、", ",")
    
    # Remove spaces around Chinese
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)
    name = re.sub(r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+])\s+(?=[\u4e00-\u9fa5])", "", name)
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name)
    
    # Remove outer quotes
    if len(name) >= 2:
        if name.startswith('"') and name.endswith('"'):
            inner = name[1:-1]
            if '"' not in inner:
                name = inner
        if name.startswith("'") and name.endswith("'"):
            inner = name[1:-1]
            if "'" not in inner:
                name = inner
        if name.startswith(""") and name.endswith("""):
            inner = name[1:-1]
            if """ not in inner and """ not in inner:
                name = inner
    
    if remove_inner_quotes:
        name = name.replace(""", "").replace(""", "").replace("'", "").replace("'", "")
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        name = name.replace("\u00a0", " ")
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)
    
    name = name.strip()
    
    # Filter pure numeric content
    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""
    
    return name


def sanitize_and_normalize_extracted_text(input_text: str, remove_inner_quotes: bool = False) -> str:
    """Sanitize and normalize extracted text."""
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        return normalize_extracted_info(safe_input_text, remove_inner_quotes=remove_inner_quotes)
    return ""


def pack_user_ass_to_openai_messages(*args: str):
    """Pack user and assistant messages for OpenAI format."""
    messages = []
    for i, content in enumerate(args):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": content})
    return messages


def remove_think_tags(text: str) -> str:
    """
    Remove thinking tags from text, including malformed tags.
    Handles various formats:
    - <think>...</think>
    - Unclosed <think> tags
    - Standalone </think> tags
    - Mixed case variations
    """
    if not text:
        return text
    
    # Remove complete think tag pairs (case insensitive)
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    if len(text.split("</think>")) > 1:
        text = text.split("</think>")[1].strip()
    else:
        text = text.strip() 
        
    return text


class Tokenizer:
    """Simple tokenizer wrapper - can be replaced with tiktoken or other"""
    
    def __init__(self):
        pass
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens (simplified - uses character count as approximation)"""
        # This is a simplified version - should use actual tokenizer
        return list(range(len(text)))
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens to text (simplified)"""
        # This is a simplified version
        return "".join(chr(i) for i in tokens if 0 <= i < 0x110000)


# Statistic tracking
statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}
