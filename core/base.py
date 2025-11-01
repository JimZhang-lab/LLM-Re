"""Base classes and interfaces for RE module."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

# This will be imported from the parent project
# For now, we define minimal versions needed

@dataclass
class StorageNamespace(ABC):
    """Base storage namespace."""
    namespace: str
    workspace: str
    global_config: dict[str, Any]

    async def initialize(self):
        """Initialize the storage"""
        pass

    async def finalize(self):
        """Finalize the storage"""
        pass

@dataclass  
class BaseKVStorage(StorageNamespace, ABC):
    """Base KV storage interface."""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get value by id"""
        pass

    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Upsert data"""
        pass

@dataclass
class BaseVectorStorage(StorageNamespace, ABC):
    """Base vector storage interface."""
    
    @abstractmethod
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vectors"""
        pass

    @abstractmethod
    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        """Query vectors"""
        pass

@dataclass
class BaseGraphStorage(StorageNamespace, ABC):
    """Base graph storage interface."""
    
    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Insert or update a node"""
        pass

    @abstractmethod
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Insert or update an edge"""
        pass
