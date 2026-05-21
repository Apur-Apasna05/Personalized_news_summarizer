"""
scheduler/jobs.py
Two-loop scheduler:

  Fast loop (every 30 min) — ingestion pipeline
    fetch → deduplicate → store raw articles

  Slow loop (every 4 hrs)  — processing pipeline
    embed → cluster → summarise → store clusters

Run with:
    python -m scheduler.jobs
or via:
    python run.py --scheduler
"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval   import IntervalTrigger

from config.settings      import FAST_LOOP_INTERVAL_MINUTES, SLOW_LOOP_INTERVAL_HOURS
from storage.database     import init_db
from ingestion.pipeline   import run_pipeline       as run_ingestion
from processing.pipeline  import run_processing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fast_loop_job():
    logger.info("Scheduler → fast loop triggered")
    run_ingestion()


def slow_loop_job():
    logger.info("Scheduler → slow loop triggered")
    run_processing_pipeline()


def start_scheduler():
    init_db()

    # Run both once on startup
    logger.info("Startup: running ingestion...")
    run_ingestion()

    logger.info("Startup: running processing...")
    run_processing_pipeline(force=True)

    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        fast_loop_job,
        trigger=IntervalTrigger(minutes=FAST_LOOP_INTERVAL_MINUTES),
        id="fast_loop",
        name="News ingestion",
        replace_existing=True,
    )
    scheduler.add_job(
        slow_loop_job,
        trigger=IntervalTrigger(hours=SLOW_LOOP_INTERVAL_HOURS),
        id="slow_loop",
        name="Embed + cluster + summarise",
        replace_existing=True,
    )

    logger.info(
        "Scheduler running — ingestion every %d min, processing every %d hrs",
        FAST_LOOP_INTERVAL_MINUTES,
        SLOW_LOOP_INTERVAL_HOURS,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start_scheduler()