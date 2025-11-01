"""
RE Core Module - Core functionality for entity-relationship extraction
"""
from __future__ import annotations

# Core data models and types
from .models import Entity, Relationship, ExtractionResult

# Configuration constants
from .constants import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_RELATIONSHIP_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ASYNC,
    DEFAULT_ENTITY_NAME_MAX_LENGTH,
)

# Extraction logic
from .extractor import (
    _process_extraction_result,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
)

# Pipeline functions
from .pipeline import (
    extract,
    extract_from_text,
    extract_from_chunks,
)

# Storage implementations
from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXGraphStorage,
)

# Base interfaces
from .base import (
    BaseKVStorage,
    BaseVectorStorage,
    BaseGraphStorage,
)

# Prompts
from .prompts import PROMPTS

# Utilities
from .utils import (
    compute_mdhash_id,
    compute_args_hash,
    generate_cache_key,
    sanitize_text_for_encoding,
    normalize_extracted_info,
    split_string_by_multi_markers,
    sanitize_and_normalize_extracted_text,
    pack_user_ass_to_openai_messages,
    remove_think_tags,
)

__all__ = [
    # Models
    "Entity",
    "Relationship",
    "ExtractionResult",
    
    # Constants
    "DEFAULT_ENTITY_TYPES",
    "DEFAULT_RELATIONSHIP_TYPES",
    "DEFAULT_SUMMARY_LANGUAGE",
    "GRAPH_FIELD_SEP",
    "DEFAULT_MAX_ASYNC",
    "DEFAULT_ENTITY_NAME_MAX_LENGTH",
    
    # Pipeline
    "extract",
    "extract_from_text",
    "extract_from_chunks",
    
    # Storage
    "JsonKVStorage",
    "NanoVectorDBStorage",
    "NetworkXGraphStorage",
    
    # Base
    "BaseKVStorage",
    "BaseVectorStorage",
    "BaseGraphStorage",
    
    # Prompts
    "PROMPTS",
    
    # Utils
    "compute_mdhash_id",
    "compute_args_hash",
    "generate_cache_key",
    "sanitize_text_for_encoding",
    "normalize_extracted_info",
    "split_string_by_multi_markers",
    "sanitize_and_normalize_extracted_text",
    "pack_user_ass_to_openai_messages",
    "remove_think_tags",
    
    # Internal functions
    "_process_extraction_result",
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
]
