import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "iarmx_colab.ipynb"


def code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def as_python(source: str) -> str:
    """Replace IPython shell (!) and magic (%) lines with `pass` at the same indent."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            line = line[: len(line) - len(stripped)] + "pass"
        lines.append(line)
    return "\n".join(lines)


def test_notebook_code_cells_are_valid_python():
    for i, source in enumerate(code_cells()):
        compile(as_python(source), f"notebook cell {i}", "exec")


def test_notebook_references_existing_files_and_trainer_settings():
    text = "\n".join(code_cells())
    for path in set(re.findall(r"\b((?:configs|scripts|tests)/[\w./-]+\.(?:yaml|py))", text)):
        assert (ROOT / path).exists(), path
    trainer = (ROOT / "iarmx" / "training" / "train.py").read_text()
    keys = set(re.findall(r"training\.(\w+)=", text))
    assert {"output_dir", "target_tokens", "init_from", "precision"} <= keys
    for key in keys:
        assert f'"{key}"' in trainer, f"notebook sets training.{key}, which the trainer never reads"


def test_notebook_run_helper_streams_logs_and_stops_on_failure(tmp_path, capsys):
    import pytest

    source = next(c for c in code_cells() if "def run(" in c)
    namespace = {}
    exec(source[source.index("import os, shlex, subprocess"):], namespace)
    log = tmp_path / "train.log"
    namespace["run"]("echo hello", log=str(log))
    assert "hello" in capsys.readouterr().out
    assert log.read_text() == "hello\n"
    with pytest.raises(RuntimeError, match="exit code 3"):
        namespace["run"]("exit 3")
