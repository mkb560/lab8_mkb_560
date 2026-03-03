import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

from week5.database import DEFAULT_DB_CONFIG
import mysql.connector


WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Doc2VecConfig:
    name: str
    vector_size: int
    min_count: int
    epochs: int
    window: int = 8
    dm: int = 1


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = WHITESPACE_RE.sub(" ", text).strip()
    tokens = [tok for tok in text.split(" ") if len(tok) >= 2]
    return tokens or ["empty"]


def load_posts_for_embedding(limit: int | None = None) -> Tuple[pd.DataFrame, str]:
    db_config = DEFAULT_DB_CONFIG.copy()
    conn = mysql.connector.connect(**db_config)
    try:
        cur = conn.cursor(dictionary=True)

        sql_cleaned = """
            SELECT post_id, subreddit, title_clean, body_clean, ocr_text
            FROM cleaned_posts
            ORDER BY created_dt DESC
        """
        if limit:
            cur.execute(sql_cleaned + " LIMIT %s", (limit,))
        else:
            cur.execute(sql_cleaned)
        cleaned_rows = cur.fetchall()

        if cleaned_rows:
            df = pd.DataFrame(cleaned_rows)
            df["combined_text"] = (
                df["title_clean"].fillna("")
                + " "
                + df["body_clean"].fillna("")
                + " "
                + df["ocr_text"].fillna("")
            ).str.strip()
            return df, "cleaned_posts"

        sql_raw = """
            SELECT post_id, subreddit, title, selftext
            FROM raw_posts
            ORDER BY created_utc DESC
        """
        if limit:
            cur.execute(sql_raw + " LIMIT %s", (limit,))
        else:
            cur.execute(sql_raw)
        raw_rows = cur.fetchall()

        if not raw_rows:
            raise RuntimeError("No rows found in cleaned_posts or raw_posts.")

        df = pd.DataFrame(raw_rows)
        df["combined_text"] = (
            df["title"].fillna("")
            + " "
            + df["selftext"].fillna("")
        ).str.strip()
        return df, "raw_posts"
    finally:
        conn.close()


def train_and_embed(post_ids: List[str], tokens_list: List[List[str]], cfg: Doc2VecConfig) -> np.ndarray:
    tagged_docs = [TaggedDocument(words=tokens_list[i], tags=[post_ids[i]]) for i in range(len(post_ids))]

    model = Doc2Vec(
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=cfg.min_count,
        workers=4,
        epochs=cfg.epochs,
        dm=cfg.dm,
        negative=10,
        seed=42,
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=len(tagged_docs), epochs=model.epochs)

    vectors = np.vstack([model.dv[pid] for pid in post_ids]).astype(np.float32)
    return vectors


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def cluster_with_cosine(vectors: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float]:
    x = l2_normalize(vectors)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    labels = model.fit_predict(x)
    sil = silhouette_score(x, labels, metric="cosine")
    return labels, float(sil)


def nearest_posts_per_cluster(
    post_ids: List[str],
    vectors: np.ndarray,
    labels: np.ndarray,
    top_n: int = 3,
) -> Dict[str, List[str]]:
    x = l2_normalize(vectors)
    out: Dict[str, List[str]] = {}

    for c in np.unique(labels):
        mask = labels == c
        cluster_vectors = x[mask]
        cluster_post_ids = np.array(post_ids)[mask]
        center = cluster_vectors.mean(axis=0, keepdims=True)
        dists = cosine_distances(cluster_vectors, center).reshape(-1)
        rank = np.argsort(dists)[:top_n]
        out[str(int(c))] = cluster_post_ids[rank].tolist()

    return out


def run_experiments(df: pd.DataFrame, n_clusters: int, output_dir: Path) -> pd.DataFrame:
    post_ids = df["post_id"].astype(str).tolist()
    tokens_list = [tokenize(t) for t in df["combined_text"].tolist()]

    configs = [
        Doc2VecConfig(name="cfg_small", vector_size=50, min_count=2, epochs=20),
        Doc2VecConfig(name="cfg_medium", vector_size=100, min_count=3, epochs=25),
        Doc2VecConfig(name="cfg_large", vector_size=200, min_count=5, epochs=35),
    ]

    summary_rows: List[Dict] = []

    for cfg in configs:
        print(f"[RUN] {cfg.name} -> vector_size={cfg.vector_size}, min_count={cfg.min_count}, epochs={cfg.epochs}")
        vectors = train_and_embed(post_ids, tokens_list, cfg)
        labels, sil = cluster_with_cosine(vectors, n_clusters=n_clusters)

        result_df = pd.DataFrame(
            {
                "post_id": post_ids,
                "subreddit": df["subreddit"].tolist(),
                "cluster_id": labels,
                "config_name": cfg.name,
                "vector_size": cfg.vector_size,
            }
        )

        result_path = output_dir / f"doc2vec_clusters_{cfg.name}.csv"
        result_df.to_csv(result_path, index=False)

        exemplars = nearest_posts_per_cluster(post_ids, vectors, labels, top_n=3)
        with open(output_dir / f"doc2vec_exemplars_{cfg.name}.json", "w", encoding="utf-8") as f:
            json.dump(exemplars, f, ensure_ascii=False, indent=2)

        row = asdict(cfg)
        row.update(
            {
                "n_docs": len(post_ids),
                "n_clusters": n_clusters,
                "silhouette_cosine": round(sil, 4),
                "cluster_size_std": round(float(pd.Series(labels).value_counts().std()), 2),
                "cluster_size_min": int(pd.Series(labels).value_counts().min()),
                "cluster_size_max": int(pd.Series(labels).value_counts().max()),
            }
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("silhouette_cosine", ascending=False)
    summary_df.to_csv(output_dir / "doc2vec_summary.csv", index=False)

    with open(output_dir / "doc2vec_best_config.json", "w", encoding="utf-8") as f:
        best = summary_df.iloc[0].to_dict()
        json.dump(best, f, ensure_ascii=False, indent=2)

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Member A - Week8 Doc2Vec experiments")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cleaned posts for quick runs")
    parser.add_argument("--k", type=int, default=8, help="Number of clusters for AgglomerativeClustering")
    parser.add_argument(
        "--out",
        type=str,
        default="week8/output_doc2vec",
        help="Output directory for csv/json results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading posts for embeddings...")
    df, source_table = load_posts_for_embedding(limit=args.limit)
    print(f"Loaded {len(df)} docs from {source_table}")

    summary_df = run_experiments(df=df, n_clusters=args.k, output_dir=out_dir)

    print("\n=== Doc2Vec Experiment Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
