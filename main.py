"""
VectorDB — Python port of the C++ VectorDB project
Implements HNSW, KD-Tree, and Brute Force search + RAG pipeline via Ollama
Run:  pip install flask requests
      python main.py
"""

import math
import time
import random
import threading
import heapq
import requests
from flask import Flask, request, jsonify, send_file
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16   # demo vector dimensions

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]

DistFn = Callable[[List[float], List[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)

def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))

def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":    return cosine
    if metric == "manhattan": return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort()
        return results[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item  = item
        self.left  = None
        self.right = None

class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root = None

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left  = self._insert(node.left,  v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: List[float], k: int, depth: int,
             dist: DistFn, heap: list):
        if node is None:
            return
        dn = dist(q, node.item.emb)
        # Max-heap (negate distance)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        ax   = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer,  q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        heap = []
        self._knn(self.root, q, k, 0, dist, heap)
        results = [(-neg_d, id_) for neg_d, id_ in heap]
        results.sort()
        return results

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    def __init__(self, item: VectorItem, max_layer: int):
        self.item      = item
        self.max_layer = max_layer
        self.neighbors: List[List[int]] = [[] for _ in range(max_layer + 1)]

class HNSW:
    def __init__(self, M: int = 16, ef_construction: int = 200):
        self.M              = M
        self.M0             = 2 * M
        self.ef_construction = ef_construction
        self.mL             = 1.0 / math.log(M)
        self.graph: Dict[int, HNSWNode] = {}
        self.entry_point    = -1
        self.top_layer      = -1
        self._rng           = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, layer: int,
                      dist: DistFn) -> List[Tuple[float, int]]:
        visited = {ep}
        d0      = dist(q, self.graph[ep].item.emb)
        # cands: min-heap (distance, id)
        # found: max-heap (neg distance, id) — to pop worst
        cands = [(d0, ep)]
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst   = -found[0][0]
            if len(found) >= ef and cd > worst:
                break
            node = self.graph.get(cid)
            if node is None or layer >= len(node.neighbors):
                continue
            for nid in node.neighbors[layer]:
                if nid in visited or nid not in self.graph:
                    continue
                visited.add(nid)
                nd = dist(q, self.graph[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-neg_d, id_) for neg_d, id_ in found]
        result.sort()
        return result

    def _select_neighbors(self, candidates: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [id_ for _, id_ in candidates[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        id_  = item.id
        lvl  = self._rand_level()
        node = HNSWNode(item, lvl)
        self.graph[id_] = node

        if self.entry_point == -1:
            self.entry_point = id_
            self.top_layer   = lvl
            return

        ep = self.entry_point
        for lc in range(self.top_layer, lvl, -1):
            ep_node = self.graph.get(ep)
            if ep_node and lc < len(ep_node.neighbors):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W    = self._search_layer(item.emb, ep, self.ef_construction, lc, dist)
            maxM = self.M0 if lc == 0 else self.M
            sel  = self._select_neighbors(W, maxM)
            # Ensure neighbor list is big enough
            while len(node.neighbors) <= lc:
                node.neighbors.append([])
            node.neighbors[lc] = sel

            for nid in sel:
                nbr = self.graph.get(nid)
                if nbr is None:
                    continue
                while len(nbr.neighbors) <= lc:
                    nbr.neighbors.append([])
                nbr.neighbors[lc].append(id_)
                if len(nbr.neighbors[lc]) > maxM:
                    # Prune: keep closest maxM
                    pairs = []
                    for c in nbr.neighbors[lc]:
                        cn = self.graph.get(c)
                        if cn:
                            pairs.append((dist(nbr.item.emb, cn.item.emb), c))
                    pairs.sort()
                    nbr.neighbors[lc] = [c for _, c in pairs[:maxM]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer   = lvl
            self.entry_point = id_

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_point == -1:
            return []
        ep = self.entry_point
        for lc in range(self.top_layer, 0, -1):
            ep_node = self.graph.get(ep)
            if ep_node and lc < len(ep_node.neighbors):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_: int):
        if id_ not in self.graph:
            return
        for node in self.graph.values():
            for layer in node.neighbors:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_point == id_:
            self.entry_point = -1
            for nid in self.graph:
                if nid != id_:
                    self.entry_point = nid
                    break
        del self.graph[id_]

    def get_info(self) -> dict:
        top = self.top_layer
        max_l = max(top + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []
        for id_, node in self.graph.items():
            nodes.append({
                "id": id_,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_layer
            })
            for lc in range(min(node.max_layer + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(node.neighbors):
                    for nid in node.neighbors[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {
            "topLayer": top,
            "nodeCount": len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

    def size(self) -> int:
        return len(self.graph)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims   = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf    = BruteForce()
        self._kdt   = KDTree(dims)
        self._hnsw  = HNSW(16, 200)
        self._lock  = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, emb: List[float], dist: DistFn) -> int:
        with self._lock:
            v = VectorItem(self._next_id, metadata, category, emb)
            self._next_id += 1
            self._store[v.id] = v
            self._bf.insert(v)
            self._kdt.insert(v)
            self._hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._bf.remove(id_)
            self._hnsw.remove(id_)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0  = time.perf_counter()
            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self._store:
                    v = self._store[id_]
                    hits.append({"id": v.id, "metadata": v.metadata,
                                 "category": v.category, "distance": round(d, 6),
                                 "embedding": v.emb})
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)
            def time_fn(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)
            return {
                "bruteforceUs": time_fn(lambda: self._bf.knn(q, k, dfn)),
                "kdtreeUs":     time_fn(lambda: self._kdt.knn(q, k, dfn)),
                "hnswUs":       time_fn(lambda: self._hnsw.knn(q, k, 50, dfn)),
                "itemCount":    len(self._store)
            }

    def all(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)

# =====================================================================
#  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
# =====================================================================

class DocumentDB:
    def __init__(self):
        self._store: Dict[int, DocItem] = {}
        self._hnsw  = HNSW(16, 200)
        self._bf    = BruteForce()
        self._lock  = threading.Lock()
        self._next_id = 1
        self._dims  = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(self._next_id, title, text, emb)
            self._next_id += 1
            self._store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, 50, cosine)
            out = []
            for d, id_ in raw:
                if id_ in self._store and d <= max_dist:
                    out.append((d, self._store[id_]))
            return out

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._hnsw.remove(id_)
            self._bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def get_dims(self) -> int:
        return self._dims

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end   = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url   = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30
            )
            if r.status_code != 200:
                return []
            data = r.json()
            return data.get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    demo_vectors = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
    ]
    for meta, cat, emb in demo_vectors:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  FLASK APP
# =====================================================================

app    = Flask(__name__)
db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)

# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

@app.route("/search")
def search():
    v_str  = request.args.get("v", "")
    k      = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo   = request.args.get("algo", "hnsw")

    try:
        q = [float(x) for x in v_str.split(",") if x]
    except ValueError:
        q = []

    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400

    return jsonify(db.search(q, k, metric, algo))

@app.route("/insert", methods=["POST"])
def insert():
    data = request.get_json(force=True, silent=True) or {}
    meta = data.get("metadata", "")
    cat  = data.get("category", "")
    emb  = data.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": id_})

@app.route("/delete/<int:id_>", methods=["DELETE"])
def delete(id_):
    ok = db.remove(id_)
    return jsonify({"ok": ok})

@app.route("/items")
def items():
    all_items = db.all()
    return jsonify([
        {"id": v.id, "metadata": v.metadata,
         "category": v.category, "embedding": v.emb}
        for v in all_items
    ])

@app.route("/benchmark")
def benchmark():
    v_str  = request.args.get("v", "")
    k      = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    try:
        q = [float(x) for x in v_str.split(",") if x]
    except ValueError:
        q = []
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    return jsonify(db.benchmark(q, k, metric))

@app.route("/hnsw-info")
def hnsw_info():
    return jsonify(db.hnsw_info())

@app.route("/stats")
def stats():
    return jsonify({
        "count":      db.size(),
        "dims":       DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics":    ["euclidean", "cosine", "manhattan"]
    })

# ── DOCUMENT + RAG ENDPOINTS ─────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    data  = request.get_json(force=True, silent=True) or {}
    title = data.get("title", "")
    text  = data.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids    = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({"error": (
                "Ollama unavailable. Install from https://ollama.com then run: "
                "ollama pull nomic-embed-text && ollama pull llama3.2"
            )}), 503
        chunk_title = (f"{title} [{i+1}/{len(chunks)}]"
                       if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})

@app.route("/doc/list")
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        result.append({
            "id":      d.id,
            "title":   d.title,
            "preview": preview,
            "words":   len(d.text.split())
        })
    return jsonify(result)

@app.route("/doc/delete/<int:id_>", methods=["DELETE"])
def doc_delete(id_):
    ok = doc_db.remove(id_)
    return jsonify({"ok": ok})

@app.route("/doc/search", methods=["POST"])
def doc_search():
    data     = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    k        = int(data.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503
    hits = doc_db.search(q_emb, k)
    return jsonify({"contexts": [
        {"id": doc.id, "title": doc.title, "distance": round(d, 4)}
        for d, doc in hits
    ]})

@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    data     = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "")
    k        = int(data.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx_parts = []
    for i, (d, doc) in enumerate(hits):
        ctx_parts.append(f"[{i+1}] {doc.title}:\n{doc.text}\n")
    context = "\n".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}\n\nAnswer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    return jsonify({
        "answer":   answer,
        "model":    ollama.gen_model,
        "contexts": [
            {"id": doc.id, "title": doc.title,
             "text": doc.text, "distance": round(d, 4)}
            for d, doc in hits
        ],
        "docCount": doc_db.size()
    })

@app.route("/status")
def status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel":      ollama.embed_model,
        "genModel":        ollama.gen_model,
        "docCount":        doc_db.size(),
        "docDims":         doc_db.get_dims(),
        "demoDims":        DIMS,
        "demoCount":       db.size()
    })

# ── SERVE FRONTEND ────────────────────────────────────────────────

@app.route("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return "index.html not found — place it in the same folder as main.py", 404
    return send_file(str(html_path))

# ── CORS (allow all origins) ──────────────────────────────────────

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>",              methods=["OPTIONS"])
def options_handler(path=""):
    return "", 204

# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    import webbrowser, threading as _t
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    # Auto-open browser after 1 second (gives Flask time to start)
    _t.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8080")).start()
    app.run(host="0.0.0.0", port=8080, threaded=True)