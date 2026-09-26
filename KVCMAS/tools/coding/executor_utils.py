import os
import json
from threading import Thread

def timeout_handler(_, __):
    raise TimeoutError()


def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)


class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):

                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret


def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()
    else:
        return result_container[0]




# Process-isolated execution: a timed-out thread running generated code cannot be killed,
# so run it in a process.
import multiprocessing as _mp
import queue as _queue


def _exec_target(q, code):
    try:
        ns = {"__name__": "__main__"}
        exec("from typing import *\n" + code, ns)
        q.put(("ok", None))
    except BaseException as e:
        q.put(("err", repr(e)))


def run_code_with_timeout(code: str, timeout: int) -> None:
    """exec `code` in a subprocess; TimeoutError on overrun, RuntimeError on raise."""
    q = _mp.Queue()
    p = _mp.Process(target=_exec_target, args=(q, code))
    p.start()
    try:
        status, err = q.get(timeout=timeout)
    except _queue.Empty:
        p.terminate(); p.join()
        raise TimeoutError()
    p.join()
    if status == "err":
        raise RuntimeError(err)


def _output_target(q, func, call):
    try:
        ns = {"__name__": "__main__"}
        exec("from typing import *\n" + func, ns)
        q.put(("ok", repr(eval(call, ns))))
    except BaseException as e:
        q.put(("err", str(e)))


def eval_output_with_timeout(func: str, call: str, timeout: int) -> str:
    """Define `func` and eval `call` in one subprocess namespace; return repr/error/TIMEOUT."""
    q = _mp.Queue()
    p = _mp.Process(target=_output_target, args=(q, func, call))
    p.start()
    try:
        status, out = q.get(timeout=timeout)
    except _queue.Empty:
        p.terminate(); p.join()
        return "TIMEOUT"
    p.join()
    return out
