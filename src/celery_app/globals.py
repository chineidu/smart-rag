import asyncio

_EVENT_LOOP: asyncio.AbstractEventLoop | None = None


def _get_worker_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create a stable event loop for the worker process.

    Celery's prefork model can leave loops closed after task retries. We reuse a
    single loop per worker and recreate it if it's ever closed.
    """
    global _EVENT_LOOP

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None and not loop.is_closed():
        return loop

    if _EVENT_LOOP is not None and not _EVENT_LOOP.is_closed():
        return _EVENT_LOOP

    _EVENT_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_EVENT_LOOP)
    return _EVENT_LOOP
