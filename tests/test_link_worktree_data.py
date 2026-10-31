"""tools/link_worktree_data.py: what a new worktree is given, and from where."""
import importlib.util
import os
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "tools" / "link_worktree_data.py"
_SPEC = importlib.util.spec_from_file_location("link_worktree_data", _PATH)
link_worktree_data = importlib.util.module_from_spec(_SPEC)
sys.modules["link_worktree_data"] = link_worktree_data
_SPEC.loader.exec_module(link_worktree_data)

ENCODER = "system/models/engine_cache/stream_s4_e60-stream_s4-encoder-b4-d3a601afe5634cc0.engine"
SHARED_ENGINE = "system/models/engine_cache/ball_detector-heatmap_s4-b4-bec16c318bac1aad.engine"
CANDIDATE = "system/models/candidates/stream_s4_e60.pt"


def _write(root: Path, rel: str, data: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _is_link(path: Path) -> bool:
    """A symlink, or (on Windows) a junction: a reparse point either way."""
    if path.is_symlink():
        return True
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    return bool(attributes & getattr(os, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _trees(tmp_path):
    main, reference, here = tmp_path / "main", tmp_path / "reference", tmp_path / "new"
    _write(main, "system/models/ball_detector.pt", b"weights")
    _write(main, SHARED_ENGINE, b"main build")
    _write(reference, SHARED_ENGINE, b"reference build")
    _write(reference, ENCODER, b"reference encoder")
    _write(reference, CANDIDATE, b"stream_s4")
    here.mkdir()
    return main, reference, here


def test_a_new_worktree_gets_the_reference_candidates_and_engines_as_copies(tmp_path):
    """The candidates the main copy lacks and the reference's engine builds reach the new tree as
    plain files: the reference's build wins over the main copy's for an engine both have, and
    the rest of the main copy's models are copied beside them."""
    main, reference, here = _trees(tmp_path)
    link_worktree_data.copy_data(here, main, reference)
    assert (here / CANDIDATE).read_bytes() == b"stream_s4"
    assert (here / ENCODER).read_bytes() == b"reference encoder"
    assert (here / SHARED_ENGINE).read_bytes() == b"reference build"
    assert (here / "system/models/ball_detector.pt").read_bytes() == b"weights"
    for rel in (*link_worktree_data.REFERENCE_DIRS, CANDIDATE, ENCODER):
        assert not _is_link(here / rel), rel
    (here / ENCODER).write_bytes(b"changed in the new tree")
    assert (reference / ENCODER).read_bytes() == b"reference encoder"


def test_an_engine_the_worktree_already_has_is_kept(tmp_path):
    main, reference, here = _trees(tmp_path)
    _write(here, ENCODER, b"own build")
    link_worktree_data.copy_data(here, main, reference)
    assert (here / ENCODER).read_bytes() == b"own build"
    assert (here / CANDIDATE).read_bytes() == b"stream_s4"


def test_the_reference_itself_and_a_missing_reference_take_the_main_copys_models(tmp_path):
    main, reference, here = _trees(tmp_path)
    link_worktree_data.copy_data(reference, main, reference)
    assert (reference / SHARED_ENGINE).read_bytes() == b"reference build"   # never overwritten
    link_worktree_data.copy_data(here, main, tmp_path / "gone")
    assert (here / SHARED_ENGINE).read_bytes() == b"main build"
    assert not (here / "system/models/candidates").exists()
