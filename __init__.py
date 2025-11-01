"""
RE (Relationship Extraction) Module - Lightweight Entity-Relationship Knowledge Graph Generation

This module extracts entity-relationship knowledge graph generation functionality 
from LightRAG into a standalone, lightweight module.

Key Features:
- Entity and relationship extraction from text
- Lightweight storage (JSON KV, NanoVectorDB, NetworkX)
- OpenAI LLM integration
- Knowledge graph merging and summarization
"""

__version__ = "1.0.0"

# Import from core module
from .core import (
    # Pipeline functions
    extract,
    extract_from_text,
    extract_from_chunks,
    
    # Storage implementations
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXGraphStorage,
    
    # Types
    Entity,
    Relationship,
    ExtractionResult,
    
    # Constants
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    GRAPH_FIELD_SEP,
    
    # Internal functions (for advanced usage)
    _process_extraction_result,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
)

__all__ = [
    # Main functions
    "extract",
    "extract_from_text",
    "extract_from_chunks",
    
    # Storage implementations
    "JsonKVStorage",
    "NanoVectorDBStorage",
    "NetworkXGraphStorage",
    
    # Types
    "Entity",
    "Relationship",
    "ExtractionResult",
    
    # Constants
    "DEFAULT_ENTITY_TYPES",
    "DEFAULT_SUMMARY_LANGUAGE",
    "GRAPH_FIELD_SEP",
    
    # Internal functions
    "_process_extraction_result",
    "_handle_single_entity_extraction",
    "_handle_single_relationship_extraction",
]
