"""RQ worker entry point: ``python -m evalforge.worker``.

Uses ``SimpleWorker`` (no fork per job) because fork-after-import is unsafe on
macOS and pointless for CPU-bound sklearn jobs with concurrency 1 on a laptop.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def main() -> None:
    from redis import Redis
    from rq import Queue, SimpleWorker

    from evalforge.config import get_settings
    from evalforge.migrations import upgrade_to_head

    logging.basicConfig(level=logging.INFO)
    upgrade_to_head()
    conn = Redis.from_url(get_settings().redis_url)
    SimpleWorker([Queue("evalforge", connection=conn)], connection=conn).work(with_scheduler=False)


if __name__ == "__main__":
    main()
