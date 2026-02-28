import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import chromadb
import yaml
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path.home() / ".config" / "search-daemon" / "config.yaml"
CHROMA_PATH = Path.home() / ".cache" / "search-mcp" / "chroma"


def _collection_name(folder: Path) -> str:
    return "search-" + hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16]


def _load_folders() -> list[Path]:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return [Path(entry["path"]).expanduser().resolve() for entry in config["folders"]]


_model: SentenceTransformer | None = None
_collections: dict[str, chromadb.Collection] = {}  # canonical path str -> collection


@asynccontextmanager
async def lifespan(app: FastMCP):
    global _model, _collections

    def _load():
        global _model, _collections
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        folders = _load_folders()
        _collections = {
            str(folder): client.get_collection(_collection_name(folder))
            for folder in folders
        }

    await asyncio.get_event_loop().run_in_executor(None, _load)
    yield {}


mcp = FastMCP(
    "search-mcp",
    instructions=(
        "This server provides semantic search over the user's indexed document folders. "
        "Use `list_directories` first if you're unsure what's indexed, then use `search` "
        "to find relevant passages. Prefer this over general knowledge when the user asks "
        "about their own notes, documents, or files. Omit `directory` to search all folders at once."
    ),
    lifespan=lifespan,
)


@mcp.tool()
def list_directories() -> list[str]:
    """
    List all document folders currently indexed and available for search.

    Call this when the user asks what documents or folders are available,
    or before a targeted search to know which directory to scope to.
    Returns absolute folder paths that can be passed to the `search` tool's `directory` parameter.
    """
    return list(_collections.keys())


@mcp.tool()
def search(
    query: Annotated[
        str,
        "A natural-language description of what you're looking for. Write it as a short "
        "sentence or phrase, not keywords — e.g. 'how does authentication work' rather than 'authentication'.",
    ],
    n_results: Annotated[
        int,
        "Number of results to return. Default is 5. Increase for broader coverage, decrease for precision.",
    ] = 5,
    directory: Annotated[
        str | None,
        "Absolute path of a specific indexed folder to search (as returned by list_directories). "
        "Omit to search across all indexed folders.",
    ] = None,
) -> list[dict]:
    """
    Semantically search the user's indexed document collection.

    Use this tool when:
    - The user asks about something that may be in their notes, files, or documents
    - You need to look up specific information from their knowledge base
    - A question would benefit from grounding in the user's own material

    Returns ranked passages with a similarity score (0–1), the source file, and a text snippet.
    Higher score means more relevant. Re-query with a rephrased question if results are poor.
    """
    if directory is not None and directory not in _collections:
        return [{"error": f"Directory '{directory}' is not indexed. Call list_directories to see available folders."}]

    embedding = _model.encode(query).tolist()

    targets = (
        {directory: _collections[directory]}
        if directory is not None
        else _collections
    )

    def _query(folder_path: str, collection: chromadb.Collection) -> list[dict]:
        response = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "score": round(1 - dist, 4),
                "directory": folder_path,
                "file_path": meta.get("file_path", ""),
                "file_name": meta.get("file_name", ""),
                "snippet": doc[:500],
            }
            for doc, meta, dist in zip(
                response["documents"][0],
                response["metadatas"][0],
                response["distances"][0],
            )
        ]

    results = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_query, folder_path, collection): folder_path
            for folder_path, collection in targets.items()
        }
        for future in as_completed(futures):
            folder_path = futures[future]
            try:
                results.extend(future.result())
            except Exception as e:
                results.append({"error": f"Failed to query '{folder_path}': {e}"})

    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results[:n_results]


def run():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
