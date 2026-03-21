"""Resolve checkpoint file paths for training/inference (single file, directory scan, or prefix)."""

import re
from pathlib import Path


_STEP_SUFFIX_RE = re.compile(r".*-(\d+)(?:\.pt)?$")


def _extract_step(path: Path) -> int:
    """Return the trailing ``-N`` step from a filename, or ``-1`` if absent."""
    match = _STEP_SUFFIX_RE.match(path.name)
    if match is None:
        return -1
    return int(match.group(1))


def latest_checkpoint(path: str | None) -> str | None:
    """Pick a ``.pt`` checkpoint: explicit file, ``path.pt`` if missing, else newest in a directory.

    In a directory, prefers the largest ``-step`` suffix; ties break by mtime.
    """
    if path is None:
        return None

    p = Path(path)
    if p.is_file():
        return str(p)

    if not p.exists():
        pt = Path(f"{path}.pt")
        if pt.is_file():
            return str(pt)
        return None

    if p.is_dir():
        candidates = []
        for item in p.iterdir():
            if not item.is_file():
                continue
            if item.suffix == ".pt" or _STEP_SUFFIX_RE.match(item.name) is not None:
                candidates.append(item)
        if not candidates:
            return None
    else:
        parent = p.parent if str(p.parent) != "" else Path(".")
        prefix = p.name
        if not parent.exists():
            return None
        candidates = [item for item in parent.iterdir() if item.is_file() and item.name.startswith(prefix)]
        if not candidates:
            pt = Path(f"{path}.pt")
            return str(pt) if pt.is_file() else None

    best = max(candidates, key=lambda item: (_extract_step(item), item.stat().st_mtime))
    return str(best)
