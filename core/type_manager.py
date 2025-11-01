"""
Type Manager for handling hierarchical entity types.
Supports coarse-fine type mapping and multi-level entity classification.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path


class TypeManager:
    """Manages hierarchical entity types with coarse-fine mapping."""
    
    def __init__(self, type_dict_path: Optional[str] = None):
        """
        Initialize TypeManager with hierarchical type dictionary.
        
        Args:
            type_dict_path: Path to coarse_fine_type_dict.json file
        """
        self.coarse_fine_mapping: Dict[str, List[str]] = {}
        self.fine_to_coarse_mapping: Dict[str, str] = {}
        self.all_types: Set[str] = set()
        
        if type_dict_path is None:
            # Default path relative to this file
            current_dir = Path(__file__).parent
            type_dict_path = current_dir.parent / "data" / "coarse_fine_type_dict.json"
        
        self.load_type_mapping(type_dict_path)
    
    def load_type_mapping(self, file_path: str) -> None:
        """Load coarse-fine type mapping from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.coarse_fine_mapping = json.load(f)
            
            # Build reverse mapping (fine -> coarse)
            for coarse_type, fine_types in self.coarse_fine_mapping.items():
                for fine_type in fine_types:
                    self.fine_to_coarse_mapping[fine_type] = coarse_type
                    self.all_types.add(fine_type)
                self.all_types.add(coarse_type)
            
            print(f"✅ Loaded {len(self.coarse_fine_mapping)} coarse types with {len(self.all_types)} total types")
            
        except FileNotFoundError:
            print(f"⚠️ Type dictionary not found at {file_path}, using empty mapping")
            self.coarse_fine_mapping = {}
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing type dictionary: {e}")
            self.coarse_fine_mapping = {}
    
    def get_coarse_types(self) -> List[str]:
        """Get all coarse-grained types."""
        return list(self.coarse_fine_mapping.keys())
    
    def get_fine_types(self, coarse_type: Optional[str] = None) -> List[str]:
        """
        Get fine-grained types.
        
        Args:
            coarse_type: If provided, return only fine types under this coarse type
            
        Returns:
            List of fine-grained types
        """
        if coarse_type is None:
            return list(self.fine_to_coarse_mapping.keys())
        
        return self.coarse_fine_mapping.get(coarse_type, [])
    
    def get_coarse_type_for_fine(self, fine_type: str) -> Optional[str]:
        """Get the coarse type for a given fine type."""
        return self.fine_to_coarse_mapping.get(fine_type)
    
    def get_all_types(self) -> List[str]:
        """Get all types (both coarse and fine)."""
        return list(self.all_types)
    
    def is_coarse_type(self, entity_type: str) -> bool:
        """Check if a type is coarse-grained."""
        return entity_type in self.coarse_fine_mapping
    
    def is_fine_type(self, entity_type: str) -> bool:
        """Check if a type is fine-grained."""
        return entity_type in self.fine_to_coarse_mapping
    
    def get_type_hierarchy(self, entity_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the hierarchy for a given entity type.
        
        Args:
            entity_type: The entity type to check
            
        Returns:
            Tuple of (coarse_type, fine_type). One will be None if the input is coarse/fine respectively.
        """
        if self.is_coarse_type(entity_type):
            return entity_type, None
        elif self.is_fine_type(entity_type):
            return self.get_coarse_type_for_fine(entity_type), entity_type
        else:
            return None, None
    
    def get_types_for_extraction(self, 
                                use_coarse_only: bool = False,
                                use_fine_only: bool = False,
                                specific_coarse_types: Optional[List[str]] = None,
                                specific_fine_types: Optional[List[str]] = None) -> List[str]:
        """
        Get types to use for entity extraction based on configuration.
        
        Args:
            use_coarse_only: If True, return only coarse types
            use_fine_only: If True, return only fine types
            specific_coarse_types: If provided, return only these coarse types and their fine types
            specific_fine_types: If provided, return only these fine types
            
        Returns:
            List of types to use for extraction
        """
        if specific_fine_types:
            return specific_fine_types
        
        if specific_coarse_types:
            types = []
            for coarse_type in specific_coarse_types:
                if coarse_type in self.coarse_fine_mapping:
                    types.append(coarse_type)
                    types.extend(self.coarse_fine_mapping[coarse_type])
            return types
        
        if use_coarse_only:
            return self.get_coarse_types()
        
        if use_fine_only:
            return self.get_fine_types()
        
        # Default: return all types
        return self.get_all_types()
    
    def format_types_for_prompt(self, 
                              types: List[str], 
                              max_types: int = None,
                              group_by_coarse: bool = True) -> str:
        """
        Format types for use in LLM prompts.
        
        Args:
            types: List of types to format
            max_types: Maximum number of types to include (None = no limit)
            group_by_coarse: If True, group fine types under their coarse types
            
        Returns:
            Formatted string for prompt
        """
        if not types:
            return "No types available"
        
        # Limit types if too many (only if max_types is specified)
        if max_types is not None and len(types) > max_types:
            types = types[:max_types]
        
        if not group_by_coarse:
            return ", ".join(sorted(types))
        
        # Group by coarse types
        grouped = {}
        for entity_type in types:
            coarse_type, fine_type = self.get_type_hierarchy(entity_type)
            
            if coarse_type and fine_type:
                # This is a fine type
                if coarse_type not in grouped:
                    grouped[coarse_type] = []
                grouped[coarse_type].append(fine_type)
            elif coarse_type:
                # This is a coarse type
                grouped[coarse_type] = []
        
        # Format grouped types
        formatted_parts = []
        for coarse_type in sorted(grouped.keys()):
            fine_types = grouped[coarse_type]
            if fine_types:
                formatted_parts.append(f"{coarse_type} ({', '.join(sorted(fine_types))})")
            else:
                formatted_parts.append(coarse_type)
        
        return ", ".join(formatted_parts)
    
    def validate_entity_type(self, entity_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if an entity type is valid and suggest corrections.
        
        Args:
            entity_type: The entity type to validate
            
        Returns:
            Tuple of (is_valid, suggested_correction)
        """
        entity_type_lower = entity_type.lower().strip()
        
        # Check exact match
        if entity_type_lower in self.all_types:
            return True, None
        
        # Check case-insensitive match
        for valid_type in self.all_types:
            if valid_type.lower() == entity_type_lower:
                return True, valid_type
        
        # Check partial match (for fine types)
        for fine_type in self.fine_to_coarse_mapping.keys():
            if entity_type_lower in fine_type.lower() or fine_type.lower() in entity_type_lower:
                return False, fine_type
        
        return False, None


# Global instance for easy access
_type_manager_instance: Optional[TypeManager] = None


def get_type_manager() -> TypeManager:
    """Get the global TypeManager instance."""
    global _type_manager_instance
    if _type_manager_instance is None:
        _type_manager_instance = TypeManager()
    return _type_manager_instance


def initialize_type_manager(type_dict_path: Optional[str] = None) -> TypeManager:
    """Initialize the global TypeManager instance."""
    global _type_manager_instance
    _type_manager_instance = TypeManager(type_dict_path)
    return _type_manager_instance
