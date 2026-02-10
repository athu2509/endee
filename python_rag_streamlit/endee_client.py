import json
import os
from typing import Any, Dict, List, Optional

import msgpack
import requests


class EndeeClient:
    """
    Minimal HTTP client for the Endee vector database.

    It talks to the OSS server you already started with ./run.sh.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        # Default to local OSS server
        self.base_url = base_url or os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
        # If NDD_AUTH_TOKEN is unset/empty, server runs in open mode and no header is needed
        self.auth_token = auth_token if auth_token is not None else (os.getenv("NDD_AUTH_TOKEN") or "")
        self.timeout = timeout

    def _headers(self, is_json: bool = True) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if is_json:
            headers["Content-Type"] = "application/json"
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers

    def health(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def create_index(self, index_name: str, dim: int, space_type: str = "cosine") -> None:
        """
        Create an index if it does not already exist.
        Safe to call repeatedly; 409 is treated as 'already exists'.
        """
        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
        }
        resp = requests.post(
            f"{self.base_url}/index/create",
            headers=self._headers(is_json=True),
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        if resp.status_code == 409:
            # Index already exists
            return
        resp.raise_for_status()

    def insert_vectors(self, index_name: str, items: List[Dict[str, Any]]) -> None:
        """
        Insert a batch of vectors.
        Each item is at minimum: {"id": str, "vector": List[float]}.
        """
        if not items:
            return

        payload = items if isinstance(items, list) else [items]
        resp = requests.post(
            f"{self.base_url}/index/{index_name}/vector/insert",
            headers=self._headers(is_json=True),
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def search(
        self,
        index_name: str,
        query_vector: List[float],
        k: int = 5,
        ef: Optional[int] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search and return the raw result dicts
        decoded from the MessagePack response.
        """
        payload: Dict[str, Any] = {
            "k": k,
            "vector": query_vector,
            "include_vectors": include_vectors,
            "include_metadata": include_metadata,
        }
        if ef is not None:
            payload["ef"] = ef

        resp = requests.post(
            f"{self.base_url}/index/{index_name}/search",
            headers=self._headers(is_json=True),
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        resp.raise_for_status()

        # Endee returns MessagePack; decode to a Python dict or list
        data = msgpack.unpackb(resp.content, raw=False)
        
        # Handle both dict and list responses
        if isinstance(data, dict):
            return data.get("results", [])
        elif isinstance(data, list):
            return data
        else:
            return []


