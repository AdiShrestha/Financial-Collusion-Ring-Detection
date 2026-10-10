"""Data-independent interface checks for the S1 learning diagnostic."""

from __future__ import annotations

import importlib.util
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SOURCE_ROOT / "scripts" / "diagnose_s1_learning.py"
CONFIG_PATH = SOURCE_ROOT / "configs" / "s1_learning_diagnostic.json"

_SPEC = importlib.util.spec_from_file_location("s1_learning_diagnostic_interface", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diagnostic
_SPEC.loader.exec_module(diagnostic)


class DiagnosticInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        diagnostic.load_helpers()

    def test_public_configuration_matches_the_declared_diagnostic_contract(self):
        config = diagnostic.strict_json(CONFIG_PATH)
        diagnostic.validate_specification(config)
        self.assertEqual(config["summary_control"]["trials"], 15)
        self.assertEqual(config["microfit"]["trials"], 60)
        self.assertNotIn("world_index", config)

    def test_optional_outputs_histories_and_references_have_no_private_defaults(self):
        parser = diagnostic.build_parser()
        args = parser.parse_args([
            "--world-index", "world_index.json",
            "--study-config", "study.json",
            "--output-dir", "results",
        ])
        self.assertIsNone(args.contract_report)
        self.assertIsNone(args.failure_history)
        self.assertIsNone(args.reference_json)

        args = parser.parse_args([
            "--world-index", "world_index.json",
            "--study-config", "study.json",
            "--output-dir", "results",
            "--contract-report", "report.md",
            "--failure-history", "failures.log",
            "--reference-json", "references.json",
        ])
        self.assertEqual(args.contract_report, Path("report.md"))
        self.assertEqual(args.failure_history, Path("failures.log"))
        self.assertEqual(args.reference_json, Path("references.json"))

    def test_reference_input_is_optional_and_hash_bound_when_supplied(self):
        self.assertEqual(diagnostic.load_descriptive_references(None), {
            "status": "NOT_PROVIDED", "values": None,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.json"
            path.write_text(json.dumps({"baseline": {"ap": 0.9}}), encoding="utf-8")
            reference = diagnostic.load_descriptive_references(path)
        self.assertEqual(reference["status"], "PROVIDED")
        self.assertEqual(reference["values"], {"baseline": {"ap": 0.9}})
        self.assertEqual(len(reference["source"]["sha256"]), 64)

    def test_source_has_no_project_bindings_or_literal_absolute_path_constructors(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        string_constants = [node.value for node in ast.walk(tree)
                            if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        self.assertFalse(any(value.startswith("project/") for value in string_constants))
        literal_paths = [node.args[0].value for node in ast.walk(tree)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == "Path"
                         and node.args and isinstance(node.args[0], ast.Constant)
                         and isinstance(node.args[0].value, str)]
        # Slash fragments in formatted trial IDs or counts are not path values.
        self.assertFalse(any(Path(value).is_absolute() for value in literal_paths))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                       for alias in node.names)
        self.assertFalse(any(name.startswith("factory.") for name in imports))


if __name__ == "__main__":
    unittest.main()
