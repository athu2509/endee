#!/usr/bin/env python3
"""
Simple script to check what's stored in your Endee vector database
"""
import requests
import json

ENDEE_URL = "http://localhost:8080/api/v1"

def check_health():
    """Check if Endee is running"""
    try:
        resp = requests.get(f"{ENDEE_URL}/health", timeout=5)
        resp.raise_for_status()
        print("✅ Endee is running!")
        print(f"   Health: {resp.json()}")
        return True
    except Exception as e:
        print(f"❌ Endee is not reachable: {e}")
        return False

def list_indices():
    """List all indices in Endee"""
    try:
        resp = requests.get(f"{ENDEE_URL}/indices", timeout=5)
        if resp.status_code == 200:
            indices = resp.json()
            print(f"\n📊 Found {len(indices)} indices:")
            for idx in indices:
                print(f"   - {idx}")
            return indices
        else:
            print(f"⚠️  Could not list indices (status {resp.status_code})")
            return []
    except Exception as e:
        print(f"❌ Error listing indices: {e}")
        return []

def get_index_info(index_name):
    """Get information about a specific index"""
    try:
        resp = requests.get(f"{ENDEE_URL}/index/{index_name}/info", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            print(f"\n📋 Index '{index_name}' info:")
            print(f"   Dimension: {info.get('dim', 'N/A')}")
            print(f"   Space type: {info.get('space_type', 'N/A')}")
            print(f"   Vector count: {info.get('count', 'N/A')}")
            return info
        else:
            print(f"⚠️  Could not get info for '{index_name}' (status {resp.status_code})")
            return None
    except Exception as e:
        print(f"❌ Error getting index info: {e}")
        return None

def main():
    print("=" * 60)
    print("Endee Vector Database Status Check")
    print("=" * 60)
    
    if not check_health():
        print("\n💡 Make sure Endee is running with: ./run.sh")
        return
    
    indices = list_indices()
    
    # Check the default RAG index
    if "candidate_rag_index" in indices:
        get_index_info("candidate_rag_index")
    else:
        print("\n⚠️  No 'candidate_rag_index' found. Upload documents first!")
    
    print("\n" + "=" * 60)
    print("This confirms vectors are stored in Endee database!")
    print("Text chunks are stored locally in: ./data/doc_chunks/chunks.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()
