"""Lightweight storage implementations for RE module"""
from __future__ import annotations
import os
import json
import time
import asyncio
from typing import Any, Optional
try:
    from .base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage, StorageNamespace
except ImportError:
    from base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage, StorageNamespace

try:
    import numpy as np
except ImportError:
    np = None

try:
    from nano_vectordb import NanoVectorDB
    NANO_AVAILABLE = True
except ImportError:
    NANO_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


def load_json(file_path: str) -> dict:
    """Load JSON from file"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def write_json(data: dict, file_path: str) -> None:
    """Write JSON to file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class SimpleLock:
    """Simple async lock for single-process usage"""
    def __init__(self):
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        await self._lock.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()


class JsonKVStorage(BaseKVStorage):
    """JSON-based key-value storage"""
    
    def __post_init__(self):
        working_dir = self.global_config.get("working_dir", "./re_workspace")
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = "_"
        
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"kv_store_{self.namespace}.json")
        self._data = {}
        self._storage_lock = SimpleLock()
    
    async def initialize(self):
        """Initialize storage"""
        if os.path.exists(self._file_name):
            self._data = load_json(self._file_name)
    
    async def index_done_callback(self) -> None:
        """Persist data to disk"""
        async with self._storage_lock:
            write_json(self._data, self._file_name)
    
    async def get_by_id(self, id: str) -> Optional[dict[str, Any]]:
        async with self._storage_lock:
            result = self._data.get(id)
            if result:
                result = dict(result)
                result.setdefault("create_time", 0)
                result.setdefault("update_time", 0)
                result["_id"] = id
            return result
    
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        async with self._storage_lock:
            results = []
            for id in ids:
                data = self._data.get(id)
                if data:
                    result = dict(data)
                    result.setdefault("create_time", 0)
                    result.setdefault("update_time", 0)
                    result["_id"] = id
                    results.append(result)
                else:
                    results.append(None)
            return results
    
    async def filter_keys(self, keys: set[str]) -> set[str]:
        async with self._storage_lock:
            return set(keys) - set(self._data.keys())
    
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        
        current_time = int(time.time())
        async with self._storage_lock:
            for k, v in data.items():
                if k in self._data:
                    v["update_time"] = current_time
                else:
                    v["create_time"] = current_time
                    v["update_time"] = current_time
                v["_id"] = k
                self._data[k] = v
    
    async def delete(self, ids: list[str]) -> None:
        async with self._storage_lock:
            for doc_id in ids:
                self._data.pop(doc_id, None)
    
    async def is_empty(self) -> bool:
        return len(self._data) == 0
    
    async def drop(self) -> dict[str, str]:
        async with self._storage_lock:
            self._data = {}
            write_json(self._data, self._file_name)
        return {"status": "success", "message": "data dropped"}


class NanoVectorDBStorage(BaseVectorStorage):
    """NanoVectorDB-based vector storage"""
    
    def __post_init__(self):
        if not NANO_AVAILABLE:
            raise ImportError("nano_vectordb is required for NanoVectorDBStorage")
        
        working_dir = self.global_config.get("working_dir", "./re_workspace")
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = "_"
        
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"vdb_{self.namespace}.json")
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim,
            storage_file=self._file_name
        )
        self._storage_lock = SimpleLock()
    
    async def initialize(self):
        pass
    
    async def index_done_callback(self) -> None:
        pass
    
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        
        current_time = int(time.time())
        list_data = [
            {"__id__": k, "__created_at__": current_time, **v}
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        
        # Execute embedding
        embeddings = await self.embedding_func(contents)
        
        async with self._storage_lock:
            for i, d in enumerate(list_data):
                d["vector"] = embeddings[i].tolist() if np is not None else list(embeddings[i])
            self._client.upsert(datas=list_data)
    
    async def query(self, query: str, top_k: int, query_embedding: list[float] = None) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = await self.embedding_func([query])
            embedding = embedding[0]
        
        async with self._storage_lock:
            results = self._client.query(query=embedding, top_k=top_k)
            return [
                {**{k: v for k, v in dp.items() if k != "vector"}, "id": dp["__id__"]}
                for dp in results
            ]
    
    async def delete(self, ids: list[str]):
        async with self._storage_lock:
            self._client.delete(ids)
    
    async def delete_entity(self, entity_name: str) -> None:
        entity_id = f"ent-{entity_name}"
        await self.delete([entity_id])
    
    async def delete_entity_relation(self, entity_name: str) -> None:
        # Implementation would need to track relations
        pass
    
    async def get_by_id(self, id: str) -> Optional[dict[str, Any]]:
        # Would need to implement retrieval from NanoVectorDB
        return None
    
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return []
    
    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        return {}
    
    async def drop(self) -> dict[str, str]:
        async with self._storage_lock:
            if os.path.exists(self._file_name):
                os.remove(self._file_name)
        return {"status": "success", "message": "data dropped"}


class NetworkXGraphStorage(BaseGraphStorage):
    """NetworkX-based graph storage"""
    
    def __post_init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required for NetworkXGraphStorage")
        
        working_dir = self.global_config.get("working_dir", "./re_workspace")
        if self.workspace:
            workspace_dir = os.path.join(working_dir, self.workspace)
        else:
            workspace_dir = working_dir
            self.workspace = "_"
        
        os.makedirs(workspace_dir, exist_ok=True)
        self._file_name = os.path.join(workspace_dir, f"graph_{self.namespace}.graphml")
        self._storage_lock = SimpleLock()
        self._graph = nx.Graph()
        
        if os.path.exists(self._file_name):
            self._graph = nx.read_graphml(self._file_name)
    
    async def initialize(self):
        pass
    
    async def index_done_callback(self) -> None:
        async with self._storage_lock:
            nx.write_graphml(self._graph, self._file_name)
    
    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)
    
    async def has_edge(self, source: str, target: str) -> bool:
        return self._graph.has_edge(source, target)
    
    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)
    
    async def edge_degree(self, src: str, tgt: str) -> int:
        src_degree = self._graph.degree(src) if self._graph.has_node(src) else 0
        tgt_degree = self._graph.degree(tgt) if self._graph.has_node(tgt) else 0
        return src_degree + tgt_degree
    
    async def get_node(self, node_id: str) -> Optional[dict[str, str]]:
        return self._graph.nodes.get(node_id)
    
    async def get_edge(self, source: str, target: str) -> Optional[dict[str, str]]:
        return self._graph.edges.get((source, target))
    
    async def get_node_edges(self, source: str) -> Optional[list[tuple[str, str]]]:
        if self._graph.has_node(source):
            return list(self._graph.edges(source))
        return None
    
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        self._graph.add_node(node_id, **node_data)
    
    async def upsert_edge(self, source: str, target: str, edge_data: dict[str, str]) -> None:
        self._graph.add_edge(source, target, **edge_data)
    
    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        results = []
        for node_id, data in self._graph.nodes(data=True):
            if "source_id" in data:
                node_chunks = set(data["source_id"].split("<SEP>"))
                if node_chunks.intersection(chunk_ids):
                    results.append(data)
        return results
    
    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        results = []
        for src, tgt, data in self._graph.edges(data=True):
            if "source_id" in data:
                edge_chunks = set(data["source_id"].split("<SEP>"))
                if edge_chunks.intersection(chunk_ids):
                    results.append(data)
        return results
    
    async def drop(self) -> dict[str, str]:
        async with self._storage_lock:
            self._graph = nx.Graph()
            if os.path.exists(self._file_name):
                os.remove(self._file_name)
        return {"status": "success", "message": "data dropped"}
