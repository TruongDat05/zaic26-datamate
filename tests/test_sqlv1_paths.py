"""Portable controller and runtime locations do not require an upstream checkout."""
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sqlv1_paths_test", ROOT / "eval/sqlv1/paths.py")
paths = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paths)


class PathsTest(unittest.TestCase):
    def test_defaults_are_sibling_controller_and_same_runtime(self):
        with patch.dict(os.environ, {}, clear=True):
            expected = ROOT.parent / "datamate_foxel"
            self.assertEqual(paths.controller_root(), expected)
            self.assertEqual(paths.upstream_root(), expected)

    def test_controller_and_snapshot_are_independent(self):
        with patch.dict(os.environ, {"DATAMATE_CONTROLLER": "./controller", "DATAMATE_UPSTREAM": "./snapshot"}, clear=True):
            self.assertEqual(paths.controller_root(), Path("controller").resolve())
            self.assertEqual(paths.upstream_root(), Path("snapshot").resolve())
            for script, key, suffix in (("audit_q35.py", "PKG", "build"),
                                        ("package_q35.py", "FOXEL", "")):
                namespace = runpy.run_path(str(ROOT / "eval/sqlv1" / script))
                self.assertEqual(namespace[key], Path("controller").resolve() / suffix)

    def test_notebook_keeps_relative_paths_resolved_at_generation(self):
        nbformat = types.ModuleType("nbformat")
        nbconvert = types.ModuleType("nbconvert")
        preprocessors = types.ModuleType("nbconvert.preprocessors")
        preprocessors.ExecutePreprocessor = object
        with tempfile.TemporaryDirectory() as directory:
            generator_cwd = Path(directory)
            kernel_cwd = generator_cwd / "kernel"
            kernel_cwd.mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(generator_cwd)
                with patch.dict(os.environ, {"DATAMATE_CONTROLLER": "./controller", "DATAMATE_UPSTREAM": "./snapshot"}, clear=True), \
                     patch.dict(sys.modules, {"nbformat": nbformat, "nbconvert": nbconvert, "nbconvert.preprocessors": preprocessors}):
                    notebook = runpy.run_path(str(ROOT / "eval/sqlv1/eda_notebook.py"))
                    cell = next(source for kind, source in notebook["CELLS"] if kind == "code" and "SNAPSHOT = Path(" in source)
                    location_code = cell[cell.index("SYNTH = Path("):cell.index("os.environ[\"DATAMATE_UPSTREAM\"]")]
                    os.chdir(kernel_cwd)
                    namespace = {"Path": Path}
                    exec(location_code, namespace)
                    self.assertEqual(namespace["FOXEL"], generator_cwd / "controller")
                    self.assertEqual(namespace["SNAPSHOT"], generator_cwd / "snapshot")
                    os.chdir(generator_cwd)
                    del os.environ["DATAMATE_UPSTREAM"]
                    notebook = runpy.run_path(str(ROOT / "eval/sqlv1/eda_notebook.py"))
                    cell = next(source for kind, source in notebook["CELLS"] if kind == "code" and "SNAPSHOT = Path(" in source)
                    location_code = cell[cell.index("SYNTH = Path("):cell.index("os.environ[\"DATAMATE_UPSTREAM\"]")]
                    os.chdir(kernel_cwd)
                    namespace = {"Path": Path}
                    exec(location_code, namespace)
                    self.assertEqual(namespace["SNAPSHOT"], generator_cwd / "controller/build/sft_v2/snapshot_8b07ba9")
            finally:
                os.chdir(old_cwd)

    def test_runtime_defaults_to_configured_controller(self):
        with patch.dict(os.environ, {"DATAMATE_CONTROLLER": "./controller"}, clear=True):
            self.assertEqual(paths.upstream_root(), Path("controller").resolve())


if __name__ == "__main__":
    unittest.main()
