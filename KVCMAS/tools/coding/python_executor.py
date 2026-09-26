import ast
import astunparse
from typing import List

from KVCMAS.tools.coding.executor_utils import function_with_timeout, run_code_with_timeout, eval_output_with_timeout
from KVCMAS.tools.coding.executor_types import ExecuteResult, Executor
import multiprocessing as mp
import textwrap
import traceback
import queue

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left               
    except:
        call_str = ast_parsed.body[0].test               

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        func_call = get_call_str(assert_statement)
        # Definition AND call run in ONE subprocess namespace with a hard timeout -- the old form
        # exec'd the definition on the MAIN thread unguarded.
        return eval_output_with_timeout(func, func_call, timeout)
    except Exception as e:
        return str(e)












def execute_code_get_return(code: str, timeout: int = 5):
    """Run ``code`` in a subprocess, kill it after ``timeout`` seconds, and return its ``answer``
    variable (or the error message). """
    def _runner(q):
        local_vars = {}
        try:

            exec(textwrap.dedent(code), {}, local_vars)
            res = local_vars.get("answer")

            if callable(res):
                res = res()

            q.put(res)
        except Exception:

            q.put(f"Error occurred:\n{traceback.format_exc()}")

    q = mp.Queue()
    p = mp.Process(target=_runner, args=(q,))
    p.start()

    try:

        result = q.get(timeout=timeout)
    except queue.Empty:
        p.terminate()
        p.join()
        return f"Timeout (> {timeout}s)"
    else:
        p.join()
        return result

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:

        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]


        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                run_code_with_timeout(func_test_list[i], timeout)
                success_tests.append(tests[i])
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests.append(f"{tests[i]} # output: {output}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """Evaluates the implementation on Human-Eval Python. """

        code = f"""{func}

{test}

check({name})
    """
        try:
            run_code_with_timeout(code, timeout)
            return True
        except Exception:
            return False

