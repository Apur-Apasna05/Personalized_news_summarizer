"""
run.py
CLI for Phase 1 + Phase 2 + Phase 3.

Commands
--------
  python run.py --ingest         Run ingestion pipeline once
  python run.py --process        Run processing pipeline once (embed+cluster+summarise)
  python run.py --sync-chroma    Push all SQLite clusters into ChromaDB
  python run.py --ask "query"    Ask a question via the RAG chain
  python run.py --rag-repl       Start an interactive RAG session
  python run.py --scheduler      Start continuous dual-loop scheduler
  python run.py --stats          Print DB stats (articles + clusters)
  python run.py --show-clusters  Print all clusters with summaries
  python run.py --chroma-stats   Print ChromaDB collection stats
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from storage.database import init_db, article_count, cluster_count, fetch_all_clusters


def cmd_ingest():
    from ingestion.pipeline import run_pipeline
    init_db()
    results = run_pipeline()
    db = results.get("db", {})
    print(f"\nIngestion complete.")
    print(f"  Total articles   : {db.get('total', '?')}")
    print(f"  Unprocessed      : {db.get('unprocessed', '?')}")


def cmd_process():
    from processing.pipeline import run_processing_pipeline
    from storage.vector_store import sync_from_db
    init_db()
    results = run_processing_pipeline(force=True)
    print(f"\nProcessing complete.")
    print(f"  Articles processed : {results.get('articles_processed', 0)}")
    print(f"  Clusters created   : {results.get('clusters_created', 0)}")
    print(f"  Total clusters     : {results.get('total_clusters', 0)}")

    # Auto-sync to ChromaDB after every process run
    print("\nSyncing clusters to ChromaDB...")
    synced = sync_from_db()
    print(f"  Synced {synced} cluster(s) to ChromaDB.")


def cmd_sync_chroma():
    from storage.vector_store import sync_from_db, collection_stats
    init_db()
    print("Syncing all clusters from SQLite → ChromaDB...")
    synced = sync_from_db()
    stats  = collection_stats()
    print(f"  Synced    : {synced} cluster(s)")
    print(f"  Chroma DB : {stats['doc_count']} total docs  ({stats['persist_dir']})")


def cmd_ask(query: str):
    from rag.chain import ask
    init_db()
    print(f"\nQ: {query}\n")
    result = ask(query)
    print(f"A: {result.answer}")
    if result.sources:
        print("\nSources used:")
        for s in result.sources:
            print(f"  • [{s['similarity']:.0%}] {s['label']}")
    else:
        print("\n(No relevant sources found in the database.)")


def cmd_rag_repl():
    from rag.chain import interactive_session
    init_db()
    interactive_session()


def cmd_chroma_stats():
    from storage.vector_store import collection_stats
    stats = collection_stats()
    print(f"\n── ChromaDB Stats ──────────────────────")
    print(f"  Collection  : {stats['collection']}")
    print(f"  Persist dir : {stats['persist_dir']}")
    print(f"  Doc count   : {stats['doc_count']}")


def cmd_stats():
    init_db()
    a = article_count()
    c = cluster_count()
    print(f"\n── DB Stats ─────────────────────────")
    print(f"  Articles total      : {a['total']}")
    print(f"  Articles unprocessed: {a['unprocessed']}")
    print(f"  Clusters total      : {c}")


def cmd_show_clusters():
    init_db()
    clusters = fetch_all_clusters()
    if not clusters:
        print("No clusters yet. Run: python run.py --process")
        return
    print(f"\n── {len(clusters)} Cluster(s) ──────────────────────")
    for cl in clusters:
        print(f"\n[{cl['id']}] {cl['label']}  ({cl['article_count']} articles)")
        summary = cl['summary']
        if summary:
            words = summary.split()
            line, lines = [], []
            for w in words:
                if sum(len(x)+1 for x in line) + len(w) > 76:
                    lines.append(" ".join(line))
                    line = []
                line.append(w)
            if line:
                lines.append(" ".join(line))
            for ln in lines:
                print(f"    {ln}")
        print(f"    Created: {cl['created_at'][:19]}")


def cmd_scheduler():
    from scheduler.jobs import start_scheduler
    start_scheduler()


def main():
    parser = argparse.ArgumentParser(description="Personalized News Summarizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest",        action="store_true",  help="Run ingestion once")
    group.add_argument("--process",       action="store_true",  help="Run processing once")
    group.add_argument("--sync-chroma",   action="store_true",  help="Sync SQLite clusters → ChromaDB")
    group.add_argument("--ask",           type=str,             metavar="QUERY", help="Ask a RAG question")
    group.add_argument("--rag-repl",      action="store_true",  help="Interactive RAG session")
    group.add_argument("--scheduler",     action="store_true",  help="Start continuous scheduler")
    group.add_argument("--stats",         action="store_true",  help="Show DB stats")
    group.add_argument("--show-clusters", action="store_true",  help="Print all clusters")
    group.add_argument("--chroma-stats",  action="store_true",  help="Show ChromaDB stats")
    args = parser.parse_args()

    if args.ingest:
        cmd_ingest()
    elif args.process:
        cmd_process()
    elif args.sync_chroma:
        cmd_sync_chroma()
    elif args.ask:
        cmd_ask(args.ask)
    elif args.rag_repl:
        cmd_rag_repl()
    elif args.scheduler:
        cmd_scheduler()
    elif args.stats:
        cmd_stats()
    elif args.show_clusters:
        cmd_show_clusters()
    elif args.chroma_stats:
        cmd_chroma_stats()


if __name__ == "__main__":
    main()
