"""Data models for the RE module."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict


class Entity(BaseModel):
    """Represents an extracted entity."""
    
    name: str = Field(description="Name of the entity")
    entity_type: str = Field(description="Type/category of the entity")
    description: str = Field(description="Description of the entity")
    source_id: str = Field(description="Source chunk ID where entity was extracted")
    file_path: str = Field(default="unknown_source", description="File path of the source")
    timestamp: int = Field(description="Timestamp when entity was extracted")
    
    # Hierarchical type information
    coarse_type: Optional[str] = Field(default=None, description="Coarse-grained type (if applicable)")
    fine_type: Optional[str] = Field(default=None, description="Fine-grained type (if applicable)")
    type_hierarchy: Optional[Dict[str, str]] = Field(default_factory=dict, description="Full type hierarchy information")
    
    # Additional metadata
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")


class Relationship(BaseModel):
    """Represents an extracted relationship between entities."""
    
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    keywords: str = Field(description="Relationship keywords")
    description: str = Field(description="Description of the relationship")
    source_id: str = Field(description="Source chunk ID where relationship was extracted")
    file_path: str = Field(default="unknown_source", description="File path of the source")
    timestamp: int = Field(description="Timestamp when relationship was extracted")
    
    # Additional metadata
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    

class ExtractionResult(BaseModel):
    """Result of entity-relationship extraction from a text chunk."""
    
    chunk_id: str = Field(description="ID of the source chunk")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Relationship] = Field(default_factory=list, description="Extracted relationships")
    timestamp: int = Field(description="Timestamp of extraction")
