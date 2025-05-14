from concurrent.futures import Future, Executor
from threading import Lock


class DummyExecutor(Executor):
    """https://stackoverflow.com/questions/10434593/dummyexecutor-for-pythons-futures

    Creates a dummy executor for regression pipelines that don't need parallelism
    or threading, and are only slowed down by the process spawning.
    """

    def __init__(self):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True
