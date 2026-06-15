"""
Agent 1: Codebase Understanding Agent
---------------------------------------
Clones a GitHub repo, chunks code/docs/specs, embeds them into a local
ChromaDB vector store, and exposes a simple query() function so other
agents (or you) can ask natural-language questions about the codebase.

Usage:
    python ingest.py --repo https://github.com/<org>/<repo>.git
    python ingest.py --test-repo https://github.com/<org>/<test-repo>.git
    python ingest.py --query "How does the login API work?"
"""

import argparse
import os
import shutil
import stat
from pathlib import Path


def _force_remove(func, path, _exc_info):
    """Error handler for shutil.rmtree: clears read-only flag then retries (needed on Windows for .git files)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKDIR = Path(__file__).parent
REPO_CACHE = WORKDIR / "repo_cache"
CHROMA_DIR = WORKDIR / "chroma_store"

CODEBASE_COLLECTION = "codebase"       # app source code
TEST_PATTERNS_COLLECTION = "test_patterns"  # existing test files

# File types we care about for understanding the app
INCLUDE_EXTENSIONS = {
    # code
    ".py", ".java", ".js", ".jsx", ".ts", ".tsx", ".go", ".rb",
    # api / config specs
    ".yaml", ".yml", ".json",
    # docs
    ".md", ".rst", ".txt",
}

# Skip noisy / irrelevant directories
EXCLUDE_DIRS = {
    ".git", "node_modules", "dist", "build", "venv", ".venv",
    "__pycache__", ".idea", ".vscode", "target", "coverage",
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Step 1: Clone (or reuse) the repo
# ---------------------------------------------------------------------------
def clone_repo(repo_url: str, force: bool = False) -> Path:
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = REPO_CACHE / repo_name

    if dest.exists() and force:
        shutil.rmtree(dest, onexc=_force_remove)

    if dest.exists():
        print(f"[ingest] Repo already cloned at {dest}, reusing.")
        return dest

    REPO_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] Cloning {repo_url} -> {dest}")
    Repo.clone_from(repo_url, dest, depth=1)  # shallow clone, faster + less disk
    return dest


# ---------------------------------------------------------------------------
# Step 2: Walk the repo and collect relevant files
# ---------------------------------------------------------------------------
def collect_files(root: Path):
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in INCLUDE_EXTENSIONS:
            continue
        # Skip very large files (likely generated/lockfiles)
        try:
            if path.stat().st_size > 500_000:
                continue
        except OSError:
            continue
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# Step 3: Chunk file contents
# ---------------------------------------------------------------------------
def chunk_files(files, root: Path):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs, metadatas, ids = [], [], []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if not text.strip():
            continue

        rel_path = str(path.relative_to(root))
        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metadatas.append({
                "source": rel_path,
                "chunk_index": i,
                "file_type": path.suffix.lstrip("."),
            })
            ids.append(f"{rel_path}::chunk{i}")

    return docs, metadatas, ids


# ---------------------------------------------------------------------------
# Step 4: Embed into ChromaDB
# ---------------------------------------------------------------------------
def build_vector_store(docs, metadatas, ids, reset: bool = False,
                       collection_name: str = CODEBASE_COLLECTION):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embed_fn,
    )

    BATCH = 200
    for i in range(0, len(docs), BATCH):
        collection.add(
            documents=docs[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
            ids=ids[i:i + BATCH],
        )

    print(f"[ingest] Indexed {len(docs)} chunks into '{collection_name}' at {CHROMA_DIR}")
    return collection


# ---------------------------------------------------------------------------
# Step 5: Query interface (used by Agent 2 and Agent 3)
# ---------------------------------------------------------------------------
def query(question: str, n_results: int = 5,
          collection_name: str = CODEBASE_COLLECTION):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(collection_name, embedding_function=embed_fn)

    results = collection.query(query_texts=[question], n_results=n_results)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"source": meta["source"], "text": doc, "distance": dist})
    return hits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Agent 1: Codebase Understanding")
    parser.add_argument("--repo", help="App GitHub repo URL to ingest into 'codebase' collection")
    parser.add_argument("--test-repo", help="Test repo URL to ingest into 'test_patterns' collection")
    parser.add_argument("--query", help="Question to ask the indexed codebase")
    parser.add_argument("--reset", action="store_true", help="Re-clone and re-index")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    if args.repo:
        repo_path = clone_repo(args.repo, force=args.reset)
        files = collect_files(repo_path)
        print(f"[ingest] Found {len(files)} relevant files in app repo")
        docs, metas, ids = chunk_files(files, repo_path)
        build_vector_store(docs, metas, ids, collection_name=CODEBASE_COLLECTION)

    if args.test_repo:
        test_repo_path = clone_repo(args.test_repo, force=args.reset)
        files = collect_files(test_repo_path)
        print(f"[ingest] Found {len(files)} relevant files in test repo")
        docs, metas, ids = chunk_files(files, test_repo_path)
        build_vector_store(docs, metas, ids, collection_name=TEST_PATTERNS_COLLECTION)

    if args.query:
        hits = query(args.query, n_results=args.top_k)
        print(f"\n[query] '{args.query}'\n" + "-" * 60)
        for h in hits:
            print(f"\nSource: {h['source']}  (distance={h['distance']:.4f})")
            print(h["text"][:400] + ("..." if len(h["text"]) > 400 else ""))

    if not args.repo and not args.test_repo and not args.query:
        parser.print_help()


if __name__ == "__main__":
    main()
