"""
Sandboxed execution of policy-generated attention programs.

Contract each candidate program must satisfy (mirrors the paper's program
signature, X -> A):

    def predict_attention(tokens: list[str]) -> np.ndarray:
        n = len(tokens)
        attention = np.zeros((n, n))
        ...
        return attention

Programs get numpy (as `np`) and `re` in scope, matching the paper's stated
tool access (NumPy / spaCy / NLTK) minus the heavier NLP libraries -- add
those back in `SAFE_GLOBALS` if your synthesis prompt advertises them, but
note each addition widens the sandbox surface and should be reviewed.

Safety notes (this is a research sandbox, not a security boundary):
  - AST-allowlists imports/calls before exec so `os`, `subprocess`, `open`,
    `eval`, `__import__`, dunder-attribute access, etc. are rejected pre-exec.
  - Runs in a subprocess with a wall-clock timeout so infinite loops / hangs
    in generated code can't stall a training run.
  - Still exec-based. If you scale this beyond a local research loop, run it
    in a real container/gVisor sandbox, not just AST filtering.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import re
import traceback
from typing import Optional

import numpy as np

FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
    "eval", "exec", "compile", "open", "__import__", "input", "exit",
    "quit", "globals", "locals", "vars", "breakpoint",
}
ALLOWED_IMPORTS = {"numpy", "re", "math", "string", "itertools", "collections"}


class UnsafeProgramError(Exception):
    pass


def static_check(code: str) -> None:
    """Raise UnsafeProgramError if the code references anything off the
    allowlist. Run this BEFORE exec, not instead of the subprocess timeout.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise UnsafeProgramError(f"SyntaxError: {e}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [mod] if mod else [a.name for a in node.names]
            for n in names:
                root = (n or "").split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    raise UnsafeProgramError(f"Disallowed import: {n}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise UnsafeProgramError(f"Disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeProgramError(f"Disallowed dunder attribute: {node.attr}")

    if "predict_attention" not in code:
        raise UnsafeProgramError("Program must define predict_attention(tokens)")


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in this sandbox")
    import importlib
    return importlib.import_module(name)


def _worker(code: str, tokens: list, out_queue: mp.Queue) -> None:
    try:
        static_check(code)
        safe_globals = {
            "__builtins__": {
                "range": range, "len": len, "enumerate": enumerate,
                "min": min, "max": max, "sum": sum, "abs": abs,
                "float": float, "int": int, "str": str, "bool": bool,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "sorted": sorted, "zip": zip, "reversed": reversed,
                "isinstance": isinstance, "print": lambda *a, **k: None,
                "__import__": _restricted_import,
            },
            "np": np,
            "numpy": np,
            "re": re,
        }
        local_ns: dict = {}
        exec(code, safe_globals, local_ns)
        fn = local_ns.get("predict_attention")
        if fn is None:
            out_queue.put(("error", "predict_attention not defined"))
            return
        result = fn(tokens)
        result = np.asarray(result, dtype=np.float64)
        n = len(tokens)
        if result.shape != (n, n):
            out_queue.put(("error", f"bad output shape {result.shape}, expected {(n, n)}"))
            return
        if not np.all(np.isfinite(result)):
            out_queue.put(("error", "non-finite values in output"))
            return
        out_queue.put(("ok", result))
    except UnsafeProgramError as e:
        out_queue.put(("error", f"unsafe: {e}"))
    except Exception:
        out_queue.put(("error", traceback.format_exc(limit=2)))


def _safe_globals() -> dict:
    return {
        "__builtins__": {
            "range": range, "len": len, "enumerate": enumerate,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "float": float, "int": int, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "sorted": sorted, "zip": zip, "reversed": reversed,
            "isinstance": isinstance, "print": lambda *a, **k: None,
            "__import__": _restricted_import,
        },
        "np": np,
        "numpy": np,
        "re": re,
    }


def _postprocess(result, n: int) -> tuple[Optional[np.ndarray], bool, Optional[str]]:
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (n, n):
        return None, False, f"bad output shape {result.shape}, expected {(n, n)}"
    if not np.all(np.isfinite(result)):
        return None, False, "non-finite values in output"
    row_sums = result.sum(axis=1, keepdims=True)
    if np.any(np.abs(row_sums - 1.0) > 1e-3):
        safe_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
        result = result / safe_sums
    return result, True, None


def compile_program(code: str):
    """Static-check `code` and exec it IN-PROCESS, returning its
    `predict_attention` callable (or raising UnsafeProgramError).

    Fast path for the RL training loop over *trusted*, template-generated
    programs (see grpo/rewarding.py): paying a fresh `spawn` subprocess per
    candidate -- the isolation `run_program` gives untrusted policy-LM output
    -- would otherwise dominate wall-clock. Do NOT feed untrusted code here.
    """
    static_check(code)
    local_ns: dict = {}
    exec(code, _safe_globals(), local_ns)
    fn = local_ns.get("predict_attention")
    if fn is None:
        raise UnsafeProgramError("predict_attention not defined")
    return fn


def run_program_inproc(code: str, tokens: list[str]
                       ) -> tuple[Optional[np.ndarray], bool, Optional[str]]:
    """In-process (no subprocess, no timeout) counterpart to `run_program`,
    for trusted code only. Same (matrix, executable, error) contract."""
    try:
        fn = compile_program(code)
        return _postprocess(fn(tokens), len(tokens))
    except UnsafeProgramError as e:
        return None, False, f"unsafe: {e}"
    except Exception:
        return None, False, traceback.format_exc(limit=2)


def run_program(code: str, tokens: list[str], timeout: float = 5.0
                ) -> tuple[Optional[np.ndarray], bool, Optional[str]]:
    """Execute `code` against `tokens` in a subprocess with a hard timeout.

    Returns (attention_matrix_or_None, executable_flag, error_or_None).
    """
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, tokens, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, False, f"timeout after {timeout}s"

    if q.empty():
        return None, False, "worker died without result (likely crashed/OOM)"

    status, payload = q.get()
    if status == "ok":
        row_sums = payload.sum(axis=1, keepdims=True)
        needs_norm = np.any(np.abs(row_sums - 1.0) > 1e-3)
        if needs_norm:
            safe_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
            payload = payload / safe_sums
        return payload, True, None
    return None, False, payload
