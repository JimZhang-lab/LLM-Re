"""Complete pipeline for entity-relationship extraction"""
from __future__ import annotations
import asyncio
import time
import logging
from typing import Any, Optional, Callable, Dict, List
from collections import defaultdict

try:
    from .constants import (
        DEFAULT_ENTITY_TYPES,
        DEFAULT_SUMMARY_LANGUAGE,
        GRAPH_FIELD_SEP,
        DEFAULT_ENTITY_NAME_MAX_LENGTH,
        DEFAULT_MAX_ASYNC,
    )
    from .models import Entity, Relationship
    from .prompts import PROMPTS
    from .utils import (
        compute_mdhash_id,
        pack_user_ass_to_openai_messages,
        remove_think_tags,
    )
    from .extractor import _process_extraction_result
except ImportError:
    from constants import (
        DEFAULT_ENTITY_TYPES,
        DEFAULT_SUMMARY_LANGUAGE,
        GRAPH_FIELD_SEP,
        DEFAULT_ENTITY_NAME_MAX_LENGTH,
        DEFAULT_MAX_ASYNC,
    )
    from models import Entity, Relationship
    from prompts import PROMPTS
    from utils import (
        compute_mdhash_id,
        pack_user_ass_to_openai_messages,
        remove_think_tags,
    )
    from extractor import _process_extraction_result

logger = logging.getLogger("lightrag.re")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


async def extract_from_text(
    text: str,
    llm_func: Callable,
    entity_types: List[str] = None,
    relationship_types: List[str] = None,
    language: str = DEFAULT_SUMMARY_LANGUAGE,
    max_gleaning: int = 1,
    tuple_delimiter: str = None,
    completion_delimiter: str = None,
    file_path: str = "unknown_source",
    # New hierarchical type parameters
    use_hierarchical_types: bool = False,
    type_mode: str = "fine",
    max_types_for_prompt: int = None,
    # Custom prompt parameters
    custom_system_prompt: str = None,
    custom_user_prompt: str = None,
    custom_continue_prompt: str = None,
    custom_examples: List[str] = None,  # Custom examples for different stages
    # History support for multi-turn conversations
    initial_history_messages: Optional[List[Dict]] = None,
) -> tuple[dict, dict]:
    """
    Extract entities and relationships from text.
    
    Args:
        text: Input text to extract from
        llm_func: LLM function to use for extraction
        entity_types: List of entity types to extract
        relationship_types: List of relationship types to extract (None = all)
        language: Language for output
        max_gleaning: Number of additional gleaning passes
        tuple_delimiter: Delimiter for tuple fields
        completion_delimiter: Delimiter for completion
        file_path: Source file path
        
    Returns:
        tuple: (entities_dict, relationships_dict)
    """
    # Set defaults and handle hierarchical types
    if use_hierarchical_types:
        from .extractor import get_entity_types_for_prompt
        entity_types_str = get_entity_types_for_prompt(
            entity_types=entity_types,
            type_mode=type_mode,
            use_hierarchical_types=True,
            max_types=max_types_for_prompt
        )
    else:
        if entity_types is None:
            entity_types = DEFAULT_ENTITY_TYPES
        entity_types_str = ", ".join(entity_types)
    
    # Prepare relationship instruction
    if relationship_types is not None and len(relationship_types) > 0:
        relationship_types_instruction = f"\n    *   **Focus on these relationship types:** {', '.join(relationship_types)}. Only extract relationships that match these types."
    else:
        relationship_types_instruction = ""
    
    if tuple_delimiter is None:
        tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    if completion_delimiter is None:
        completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    
    # Prepare prompts
    # Use custom examples if provided, otherwise use defaults
    if custom_examples:
        examples = "\n".join(custom_examples)
    else:
        examples = "\n".join(PROMPTS.get("entity_extraction_examples", []))
    
    example_context_base = dict(
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=entity_types_str,
        language=language,
    )
    
    # Format examples (if they contain placeholders)
    if examples and "{" in examples:
        try:
            examples = examples.format(**example_context_base)
        except KeyError:
            # If formatting fails, use examples as-is
            pass
    
    context_base = dict(
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=entity_types_str,
        # relationship_types=", ".join(relationship_types),
        examples=examples,
        language=language,
    )
    
    # Generate chunk key
    chunk_key = compute_mdhash_id(text, prefix="chunk-")
    timestamp = int(time.time())
    
    # Prepare prompts - use custom prompts if provided, otherwise use defaults
    if custom_system_prompt:
        try:
            entity_extraction_system_prompt = custom_system_prompt.format(
                **{**context_base, "input_text": text}
            )
        except KeyError as e:
            logger.warning(f"Custom system prompt requires missing parameter: {e}. Using default prompt.")
            entity_extraction_system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
                **{**context_base, "input_text": text}
            )
    else:
        entity_extraction_system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
            **{**context_base, "input_text": text}
        )
    
    if custom_user_prompt:
        try:
            entity_extraction_user_prompt = custom_user_prompt.format(
                **{**context_base, "input_text": text}
            )
        except KeyError as e:
            logger.warning(f"Custom user prompt requires missing parameter: {e}. Using default prompt.")
            entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
                **{**context_base, "input_text": text}
            )
    else:
        entity_extraction_user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            **{**context_base, "input_text": text}
        )
    
    if custom_continue_prompt:
        try:
            # Try to format with available context
            entity_continue_extraction_user_prompt = custom_continue_prompt.format(
                **{**context_base, "input_text": text}
            )
        except KeyError as e:
            # If custom template requires parameters we don't have, use default instead
            logger.warning(f"Custom continue prompt requires missing parameter: {e}. Using default prompt.")
            entity_continue_extraction_user_prompt = PROMPTS.get(
                "entity_continue_extraction_user_prompt", ""
            ).format(**{**context_base, "input_text": text})
    else:
        entity_continue_extraction_user_prompt = PROMPTS.get(
            "entity_continue_extraction_user_prompt", ""
        ).format(**{**context_base, "input_text": text})
    
    # Call LLM for initial extraction
    logger.info(f"Extracting from text (chunk: {chunk_key[:12]}...)")
    
    try:
        # First extraction - use initial history if provided
        if initial_history_messages:
            # If initial history is provided, use it directly
            final_result = await llm_func(
                entity_extraction_user_prompt,
                system_prompt=entity_extraction_system_prompt,
                history_messages=initial_history_messages,
            )
        else:
            # Normal first extraction
            final_result = await llm_func(
                entity_extraction_user_prompt,
                system_prompt=entity_extraction_system_prompt,
            )
        
        # Remove think tags from LLM response
        final_result = remove_think_tags(final_result)
        
        # Process initial extraction
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result,
            chunk_key,
            timestamp,
            file_path,
            tuple_delimiter=tuple_delimiter,
            completion_delimiter=completion_delimiter,
            use_hierarchical_types=use_hierarchical_types,
        )
        
        # Gleaning passes
        if max_gleaning > 0 and entity_continue_extraction_user_prompt:
            # Build history - if initial_history was provided, extend it
            if initial_history_messages:
                history = list(initial_history_messages)
                # Add current round to history
                history.extend(pack_user_ass_to_openai_messages(
                    entity_extraction_user_prompt, final_result
                ))
            else:
                history = pack_user_ass_to_openai_messages(
                    entity_extraction_user_prompt, final_result
                )
            
            # Track progress to detect infinite loops
            # Use (name, type) signatures to properly track entities with different types
            def get_entity_signatures(nodes_dict):
                """Get set of (name, type) signatures for all entities"""
                signatures = set()
                for name, entity_list in nodes_dict.items():
                    for entity in entity_list:
                        entity_type = entity.get("entity_type", "unknown")
                        signatures.add((name.lower(), entity_type.lower()))
                return signatures
            
            previous_round_entities = get_entity_signatures(maybe_nodes)
            no_new_entity_rounds = 0
            max_no_progress_rounds = 2
            
            for glean_pass in range(max_gleaning):
                logger.info(f"Gleaning pass {glean_pass + 1}/{max_gleaning}")
                
                glean_result = await llm_func(
                    entity_continue_extraction_user_prompt,
                    system_prompt=entity_extraction_system_prompt,
                    history_messages=history,
                )
                
                # Remove think tags from gleaning response
                glean_result = remove_think_tags(glean_result)
                
                # Check if response indicates no more entities
                glean_result_upper = glean_result.upper().strip()
                if "NONE" in glean_result_upper or not glean_result.strip():
                    logger.info(f"No more entities found in gleaning pass {glean_pass + 1}, stopping")
                    break
                
                # Process gleaning result
                glean_nodes, glean_edges = await _process_extraction_result(
                    glean_result,
                    chunk_key,
                    timestamp,
                    file_path,
                    tuple_delimiter=tuple_delimiter,
                    completion_delimiter=completion_delimiter,
                    use_hierarchical_types=use_hierarchical_types,
                )
                
                # Track new entities found in this round
                new_entities_this_round = 0
                current_round_entities = set(maybe_nodes.keys())
                
                # Merge results - consider entity types to allow same name with different types
                for entity_name, glean_entities in glean_nodes.items():
                    if entity_name in maybe_nodes:
                        # Check if any glean entity has a different type than existing ones
                        existing_entities = maybe_nodes[entity_name]
                        existing_types = {e.get("entity_type", "").lower() for e in existing_entities}
                        
                        entities_to_add = []
                        for glean_entity in glean_entities:
                            glean_type = glean_entity.get("entity_type", "").lower()
                            
                            if glean_type in existing_types:
                                # Same type exists, compare descriptions and update if better
                                for i, existing in enumerate(existing_entities):
                                    if existing.get("entity_type", "").lower() == glean_type:
                                        original_desc_len = len(existing.get("description", "") or "")
                                        glean_desc_len = len(glean_entity.get("description", "") or "")
                                        
                                        if glean_desc_len > original_desc_len:
                                            existing_entities[i] = glean_entity
                                        break
                            else:
                                # Different type, add as new entity variant
                                entities_to_add.append(glean_entity)
                                new_entities_this_round += 1
                        
                        # Add new type variants
                        if entities_to_add:
                            maybe_nodes[entity_name].extend(entities_to_add)
                            logger.info(f"Added {len(entities_to_add)} new type variant(s) for entity '{entity_name}'")
                    else:
                        maybe_nodes[entity_name] = list(glean_entities)
                        new_entities_this_round += len(glean_entities)
                
                for edge_key, glean_edge_list in glean_edges.items():
                    if edge_key in maybe_edges:
                        # Compare and keep better version
                        original_desc_len = len(
                            maybe_edges[edge_key][0].get("description", "") or ""
                        )
                        glean_desc_len = len(glean_edge_list[0].get("description", "") or "")
                        
                        if glean_desc_len > original_desc_len:
                            maybe_edges[edge_key] = list(glean_edge_list)
                    else:
                        maybe_edges[edge_key] = list(glean_edge_list)
                
                # Check for progress using entity signatures (name + type)
                new_round_entities = get_entity_signatures(maybe_nodes)
                
                # Detect infinite loop: same entities (name+type combinations) returned
                if new_round_entities == previous_round_entities:
                    logger.warning(f"No new entities in gleaning pass {glean_pass + 1}, detected potential loop")
                    no_new_entity_rounds += 1
                    
                    if no_new_entity_rounds >= max_no_progress_rounds:
                        logger.info(f"No progress for {no_new_entity_rounds} consecutive rounds, stopping gleaning")
                        break
                else:
                    no_new_entity_rounds = 0
                    new_signatures = new_round_entities - previous_round_entities
                    logger.info(f"Found {len(new_signatures)} new entity-type combinations in gleaning pass {glean_pass + 1}")
                
                previous_round_entities = new_round_entities
                
                # Update history for next pass
                # Extend history with the new user prompt and assistant response
                history.extend(pack_user_ass_to_openai_messages(
                    entity_continue_extraction_user_prompt, glean_result
                ))
        
        logger.info(f"Extracted {len(maybe_nodes)} entities and {len(maybe_edges)} relationships")
        
        return maybe_nodes, maybe_edges
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


async def extract_from_chunks(
    chunks: Dict[str, str],
    llm_func: Callable,
    entity_types: List[str] = None,
    relationship_types: List[str] = None,
    language: str = DEFAULT_SUMMARY_LANGUAGE,
    max_gleaning: int = 1,
    max_concurrent: int = DEFAULT_MAX_ASYNC,
    file_path: str = "unknown_source",
) -> tuple[dict, dict]:
    """
    Extract entities and relationships from multiple chunks.
    
    Args:
        chunks: Dictionary of {chunk_id: chunk_content}
        llm_func: LLM function to use
        entity_types: List of entity types to extract
        relationship_types: List of relationship types to extract (None = all)
        language: Language for output
        max_gleaning: Number of additional gleaning passes
        max_concurrent: Maximum concurrent extractions
        file_path: Source file path
        
    Returns:
        tuple: (all_entities_dict, all_relationships_dict)
    """
    all_entities = defaultdict(list)
    all_relationships = defaultdict(list)
    
    # Process chunks with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_chunk(chunk_id: str, content: str):
        async with semaphore:
            try:
                entities, relationships = await extract_from_text(
                    text=content,
                    llm_func=llm_func,
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                    language=language,
                    max_gleaning=max_gleaning,
                    file_path=file_path,
                )
                
                # Merge results
                for entity_name, entity_list in entities.items():
                    all_entities[entity_name].extend(entity_list)
                
                for edge_key, edge_list in relationships.items():
                    all_relationships[edge_key].extend(edge_list)
                
                logger.info(f"Processed chunk {chunk_id[:12]}... ({len(entities)} entities, {len(relationships)} relations)")
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_id}: {e}")
    
    # Process all chunks
    tasks = [process_chunk(chunk_id, content) for chunk_id, content in chunks.items()]
    await asyncio.gather(*tasks)
    
    logger.info(f"Total: {len(all_entities)} unique entities, {len(all_relationships)} unique relationships")
    
    return dict(all_entities), dict(all_relationships)


# Convenience function for simple usage
async def extract(text: str, llm_func: Callable, **kwargs) -> tuple[dict, dict]:
    """
    Simple extraction function.
    
    Args:
        text: Input text
        llm_func: LLM function
        **kwargs: Additional arguments passed to extract_from_text
        
    Returns:
        tuple: (entities_dict, relationships_dict)
    """
    return await extract_from_text(text, llm_func, **kwargs)
