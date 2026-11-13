"""Preflight for generated analytical code; not a general secure Python sandbox.

Reject I/O and reflective execution before running provider output locally. The
inference sandbox contract is unchanged. Only synthetic T is passed to T4 code.
"""
import ast
import ctypes
import errno
import os
from pathlib import Path
import sys

ALLOWED_IMPORTS = {"pandas", "numpy", "datetime", "calendar", "math", "re", "collections", "decimal", "itertools", "functools", "statistics"}
DENIED_CALLS = {"exec", "eval", "compile", "open", "__import__", "getattr", "setattr", "delattr", "vars", "globals", "locals", "input", "breakpoint"}


def child_environment():
    """Do not pass provider credentials or other ambient secrets to the child."""
    return {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}}


def restrict_process():
    """Install Linux filesystem and syscall restrictions, or fail closed.

    Called only in a disposable child after trusted input and imports are loaded.
    This is defense in depth for synthetic evaluation, not a Python sandbox.
    """
    if sys.platform != "linux":
        raise RuntimeError("synthetic isolation requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    if syscall(444, 0, 0, 1) < 4:
        raise RuntimeError("synthetic isolation requires Landlock ABI 4")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    handled = (1 << 15) - 1  # ABI 4, including REFER and TRUNCATE
    ruleset = Ruleset(handled)
    fd = syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Landlock ruleset")
    try:
        forbidden = [Path(__file__).resolve().parents[1], Path.home().resolve()]
        paths = {Path(p).resolve() for p in ("/usr", "/lib", "/lib64", str(Path(sys.prefix) / "lib"), str(Path(sys.base_prefix) / "lib")) if Path(p).exists()}
        for path in paths:
            if any(path == p or path in p.parents for p in forbidden):
                raise RuntimeError("runtime library path contains protected directory")
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathBeneath((1 << 0) | (1 << 2) | (1 << 3), path_fd)
                if syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "Landlock path rule")
            finally:
                os.close(path_fd)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or syscall(446, fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "Landlock restrict")
    finally:
        os.close(fd)

    sec = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint32]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = sec.seccomp_init(0x7fff0000)  # allow, except explicitly denied calls
    if not ctx:
        raise RuntimeError("seccomp initialization failed")
    try:
        if sec.seccomp_attr_set(ctx, 4, 1) != 0:  # SCMP_FLTATR_CTL_TSYNC
            raise RuntimeError("seccomp TSYNC unavailable")
        for name in ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg", "recvfrom", "recvmsg", "clone", "clone3", "fork", "vfork", "execve", "execveat", "ptrace", "process_vm_readv", "process_vm_writev", "pidfd_getfd", "io_uring_setup", "bpf", "unshare", "setns"):
            nr = sec.seccomp_syscall_resolve_name(name.encode())
            if nr >= 0 and sec.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, nr, 0) != 0:
                raise RuntimeError("seccomp rule failed: " + name)
        if sec.seccomp_load(ctx) != 0:
            raise RuntimeError("seccomp load failed")
    finally:
        sec.seccomp_release(ctx)


def preflight(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        return "invalid_python"
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.split('.')[0] not in ALLOWED_IMPORTS for a in n.names):
            return "external_import"
        if isinstance(n, ast.ImportFrom) and (n.level or (n.module or '').split('.')[0] not in ALLOWED_IMPORTS):
            return "external_import"
        if isinstance(n, ast.Name) and n.id.startswith('__'):
            return "reflection"
        if isinstance(n, ast.Attribute) and (n.attr.startswith('__') or n.attr.startswith('read_') or n.attr in {'to_csv', 'to_json', 'to_pickle', 'to_sql', 'to_excel', 'save', 'load', 'dump', 'dumps', 'system', 'popen'}):
            return "external_io_or_reflection"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in DENIED_CALLS:
            return "external_execution"
    return None
