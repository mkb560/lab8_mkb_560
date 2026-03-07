import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from week5.database import DEFAULT_DB_CONFIG
import mysql.connector

WHITESPACE_RE = re.compile(r"\s+")

@dataclass
class W2VBinConfig:
    name: str
    k_bins: int
    w2v_dim: int = 100
    min_count: int = 2
    epochs: int = 10
    window: int = 5

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = WHITESPACE_RE.sub(" ", text).strip()
    tokens = [tok for tok in text.split(" ") if len(tok) >= 2]
    return tokens or ["empty"]

def load_posts(limit: int | None = None) -> Tuple[pd.DataFrame, str]:
    db_config = DEFAULT_DB_CONFIG.copy()
    conn = mysql.connector.connect(**db_config)
    try:
        cur = conn.cursor(dictionary=True)
        sql_cleaned = """
            SELECT post_id, subreddit, title_clean, body_clean, ocr_text
            FROM cleaned_posts ORDER BY created_dt DESC
        """
        cur.execute(sql_cleaned + (" LIMIT %s" % limit if limit else ""))
        cleaned_rows = cur.fetchall()

        if cleaned_rows:
            df = pd.DataFrame(cleaned_rows)
            df["combined_text"] = (
                df["title_clean"].fillna("") + " " + 
                df["body_clean"].fillna("") + " " + 
                df["ocr_text"].fillna("")
            ).str.strip()
            return df, "cleaned_posts"

        sql_raw = "SELECT post_id, subreddit, title, selftext FROM raw_posts ORDER BY created_utc DESC"
        cur.execute(sql_raw + (" LIMIT %s" % limit if limit else ""))
        raw_rows = cur.fetchall()

        if not raw_rows:
            raise RuntimeError("No rows found in DB.")

        df = pd.DataFrame(raw_rows)
        df["combined_text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).str.strip()
        return df, "raw_posts"
    finally:
        conn.close()

def build_w2v_and_bin(tokens_list: List[List[str]], cfg: W2VBinConfig) -> Dict[str, int]:
    print(f"  -> Training Word2Vec (dim={cfg.w2v_dim})...")
    w2v_model = Word2Vec(
        sentences=tokens_list, vector_size=cfg.w2v_dim, window=cfg.window,
        min_count=cfg.min_count, workers=4, epochs=cfg.epochs, seed=42
    )
    
    words = list(w2v_model.wv.index_to_key)
    word_vectors = w2v_model.wv.vectors
    
    print(f"  -> Running KMeans to create {cfg.k_bins} bins...")
    kmeans = KMeans(n_clusters=cfg.k_bins, random_state=42, n_init=10)
    word_labels = kmeans.fit_predict(word_vectors)
    
    return dict(zip(words, word_labels))

def build_doc_vectors(tokens_list: List[List[str]], word_to_bin: Dict[str, int], k_bins: int) -> np.ndarray:
    doc_vectors = []
    for tokens in tokens_list:
        vec = np.zeros(k_bins)
        valid_words = 0
        for word in tokens:
            if word in word_to_bin:
                vec[word_to_bin[word]] += 1
                valid_words += 1
        
        if valid_words > 0:
            vec = vec / valid_words
        doc_vectors.append(vec)
        
    return np.array(doc_vectors, dtype=np.float32)

def l2_normalize(x: np.ndarray) -> np.ndarray:
    zero_rows = np.all(x == 0, axis=1)
    x[zero_rows] = 1e-5 
    
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

def cluster_with_cosine(vectors: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float]:
    x = l2_normalize(vectors)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    labels = model.fit_predict(x)
    sil = silhouette_score(x, labels, metric="cosine")
    return labels, float(sil)

def get_exemplars(post_ids: List[str], vectors: np.ndarray, labels: np.ndarray, top_n: int = 3) -> Dict[str, List[str]]:
    x = l2_normalize(vectors)
    out: Dict[str, List[str]] = {}
    for c in np.unique(labels):
        mask = labels == c
        c_vecs = x[mask]
        c_ids = np.array(post_ids)[mask]
        center = c_vecs.mean(axis=0, keepdims=True)
        dists = cosine_distances(c_vecs, center).reshape(-1)
        rank = np.argsort(dists)[:top_n]
        out[str(int(c))] = c_ids[rank].tolist()
    return out

def run_experiments(df: pd.DataFrame, n_clusters: int, out_dir: Path) -> pd.DataFrame:
    post_ids = df["post_id"].astype(str).tolist()
    tokens_list = [tokenize(t) for t in df["combined_text"].tolist()]

    configs = [
        W2VBinConfig(name="cfg_small", k_bins=50),
        W2VBinConfig(name="cfg_medium", k_bins=100),
        W2VBinConfig(name="cfg_large", k_bins=200),
    ]

    summary_rows = []

    for cfg in configs:
        print(f"\n[RUN] {cfg.name} -> k_bins={cfg.k_bins}")
        
        word_to_bin = build_w2v_and_bin(tokens_list, cfg)
        vectors = build_doc_vectors(tokens_list, word_to_bin, cfg.k_bins)
        labels, sil = cluster_with_cosine(vectors, n_clusters=n_clusters)

        res_df = pd.DataFrame({
            "post_id": post_ids,
            "subreddit": df["subreddit"].tolist(),
            "cluster_id": labels,
            "config_name": cfg.name,
            "vector_dimension": cfg.k_bins,
        })
        res_df.to_csv(out_dir / f"w2v_bin_clusters_{cfg.name}.csv", index=False)

        exemplars = get_exemplars(post_ids, vectors, labels, top_n=3)
        with open(out_dir / f"w2v_bin_exemplars_{cfg.name}.json", "w", encoding="utf-8") as f:
            json.dump(exemplars, f, ensure_ascii=False, indent=2)

        row = asdict(cfg)
        row.update({
            "n_docs": len(post_ids),
            "n_clusters": n_clusters,
            "silhouette_cosine": round(sil, 4),
            "cluster_size_std": round(float(pd.Series(labels).value_counts().std()), 2)
        })
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("silhouette_cosine", ascending=False)
    summary_df.to_csv(out_dir / "w2v_bin_summary.csv", index=False)

    with open(out_dir / "w2v_bin_best_config.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.iloc[0].to_dict(), f, ensure_ascii=False, indent=2)

    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", type=str, default="output_w2v_bin")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading posts from MySQL database...")
    df, src = load_posts(limit=args.limit)
    print(f"Loaded {len(df)} docs from {src}")

    summary_df = run_experiments(df, args.k, out_dir)

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs saved to: {out_dir}")

if __name__ == "__main__":
    main()