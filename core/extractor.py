"""Complete extraction logic for RE module - adapted from operate.py"""
from __future__ import annotations
import asyncio
import time
import logging
from typing import Any, Optional, Callable, List
from collections import defaultdict

try:
    from .constants import (
        DEFAULT_ENTITY_TYPES,
        DEFAULT_SUMMARY_LANGUAGE,
        GRAPH_FIELD_SEP,
        DEFAULT_ENTITY_NAME_MAX_LENGTH,
        DEFAULT_USE_HIERARCHICAL_TYPES,
        DEFAULT_TYPE_MODE,
        DEFAULT_MAX_TYPES_FOR_PROMPT,
    )
    from .models import Entity, Relationship
    from .prompts import PROMPTS
    from .type_manager import get_type_manager
    from .utils import (
        compute_mdhash_id,
        compute_args_hash,
        generate_cache_key,
        split_string_by_multi_markers,
        sanitize_and_normalize_extracted_text,
        pack_user_ass_to_openai_messages,
        remove_think_tags,
        Tokenizer,
    )
except ImportError:
    from constants import (
        DEFAULT_ENTITY_TYPES,
        DEFAULT_SUMMARY_LANGUAGE,
        GRAPH_FIELD_SEP,
        DEFAULT_ENTITY_NAME_MAX_LENGTH,
        DEFAULT_USE_HIERARCHICAL_TYPES,
        DEFAULT_TYPE_MODE,
        DEFAULT_MAX_TYPES_FOR_PROMPT,
    )
    from models import Entity, Relationship
    from prompts import PROMPTS
    from type_manager import get_type_manager
    from utils import (
        compute_mdhash_id,
        compute_args_hash,
        generate_cache_key,
        split_string_by_multi_markers,
        sanitize_and_normalize_extracted_text,
        pack_user_ass_to_openai_messages,
        remove_think_tags,
        Tokenizer,
    )

logger = logging.getLogger("lightrag.re")
# Configure logger if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _truncate_entity_identifier(identifier: str, limit: int, chunk_key: str, identifier_role: str) -> str:
    """Truncate entity identifiers that exceed the configured length limit"""
    if len(identifier) <= limit:
        return identifier
    
    display_value = identifier[:limit]
    preview = identifier[:20]
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    use_hierarchical_types: bool = DEFAULT_USE_HIERARCHICAL_TYPES,
):
    """Handle extraction of a single entity with hierarchical type support"""
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 fields on ENTITY `{record_attributes[1]}`"
            )
        return None
    
    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        
        if not entity_name or not entity_name.strip():
            logger.warning(f"Entity extraction error: entity name became empty")
            return None
        
        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )
        
        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(f"Entity extraction error: invalid entity type")
            return None
        
        entity_type = entity_type.replace(" ", "").lower()
        
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])
        
        if not entity_description.strip():
            logger.warning(f"Entity extraction error: empty description")
            return None
        
        # Process hierarchical type information
        entity_data = dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )
        
        if use_hierarchical_types:
            type_manager = get_type_manager()
            coarse_type, fine_type = type_manager.get_type_hierarchy(entity_type)
            
            entity_data.update({
                "coarse_type": coarse_type,
                "fine_type": fine_type,
                "type_hierarchy": {
                    "coarse": coarse_type,
                    "fine": fine_type,
                    "original": entity_type
                }
            })
            
            # Validate and potentially correct the entity type
            is_valid, suggested_type = type_manager.validate_entity_type(entity_type)
            if not is_valid and suggested_type:
                logger.info(f"Entity type '{entity_type}' corrected to '{suggested_type}'")
                entity_data["entity_type"] = suggested_type
                # Update hierarchy with corrected type
                coarse_type, fine_type = type_manager.get_type_hierarchy(suggested_type)
                entity_data.update({
                    "coarse_type": coarse_type,
                    "fine_type": fine_type,
                    "type_hierarchy": {
                        "coarse": coarse_type,
                        "fine": fine_type,
                        "original": entity_type,
                        "corrected": suggested_type
                    }
                })
        
        return entity_data
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return None


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    """Handle extraction of a single relationship"""
    if len(record_attributes) != 5 or "relation" not in record_attributes[0]:
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on RELATION"
            )
        return None
    
    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )
        
        if not source or not target:
            return None
        
        if source == target:
            return None
        
        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")
        
        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])
        
        return dict(
            src_id=source,
            tgt_id=target,
            weight=1.0,
            description=edge_description,
            keywords=edge_keywords,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )
    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        return None


async def _process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
    use_hierarchical_types: bool = DEFAULT_USE_HIERARCHICAL_TYPES,
) -> tuple[dict, dict]:
    """Process a single extraction result"""
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    
    if completion_delimiter not in result:
        logger.warning(f"{chunk_key}: Complete delimiter not found")
    
    # Split into records
    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )
    
    for record in records:
        record = record.strip()
        if not record:
            continue
        
        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])
        
        # Try to parse as entity
        entity_data = await _handle_single_entity_extraction(
            record_attributes, chunk_key, timestamp, file_path, use_hierarchical_types
        )
        if entity_data is not None:
            truncated_name = _truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            continue
        
        # Try to parse as relationship
        relationship_data = await _handle_single_relationship_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if relationship_data is not None:
            truncated_source = _truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            truncated_target = _truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)
    
    return dict(maybe_nodes), dict(maybe_edges)


def get_entity_types_for_prompt(
    entity_types: Optional[List[str]] = None,
    type_mode: str = DEFAULT_TYPE_MODE,
    use_hierarchical_types: bool = DEFAULT_USE_HIERARCHICAL_TYPES,
    max_types: int = DEFAULT_MAX_TYPES_FOR_PROMPT,
) -> str:
    """
    Get formatted entity types for use in LLM prompts.
    
    Args:
        entity_types: Custom entity types to use (overrides defaults)
        type_mode: Type extraction mode ("coarse", "fine", "both", "auto")
        use_hierarchical_types: Whether to use hierarchical type system
        max_types: Maximum number of types to include (None = no limit)
        
    Returns:
        Formatted string of entity types for prompt
    """
    if entity_types is not None:
        # Use custom entity types
        if max_types is not None and len(entity_types) > max_types:
            entity_types = entity_types[:max_types]
        return ", ".join(sorted(entity_types))
    
    if not use_hierarchical_types:
        # Use default flat types
        return ", ".join(DEFAULT_ENTITY_TYPES)
    
    # Use hierarchical types
    type_manager = get_type_manager()
    
    if type_mode == "coarse":
        types = type_manager.get_coarse_types()
    elif type_mode == "fine":
        types = type_manager.get_fine_types()
    elif type_mode == "both":
        types = type_manager.get_all_types()
    elif type_mode == "auto":
        # Auto mode: use fine types if available, otherwise coarse
        fine_types = type_manager.get_fine_types()
        # If max_types is None, always use fine types; otherwise check limit
        if max_types is None or (fine_types and len(fine_types) <= max_types):
            types = fine_types
        else:
            types = type_manager.get_coarse_types()
    else:
        types = type_manager.get_all_types()
    
    return type_manager.format_types_for_prompt(
        types, 
        max_types=max_types, 
        group_by_coarse=(type_mode != "fine")
    )


# Note: Full implementation of extract_entities and merge_nodes_and_edges
# would require adaptation from operate.py lines 2642-2884 and 2272-2640
# and integration with LLM functions and storage backends
