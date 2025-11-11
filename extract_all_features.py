#!/usr/bin/env python3
"""
extract_all_features.py

Extract all 6 DisVoice feature families from a folder of WAVs, after
auto-converting every file to 16 kHz mono. For each family, save both
CSV and JSON tables.

Families:
- glottal, phonation, articulation, prosody, phonological, replearning

Outputs:
- features_out/out_<family>.csv
- features_out/out_<family>.json

Python 3.13.9 • Ubuntu • VS Code
"""

from __future__ import annotations

import argparse
import sys
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional
import os

# --- I/O & audio ---
import numpy as np

try:
    import soundfile as sf
except Exception as e:
    print("[FATAL] soundfile is required for writing normalized WAVs:", e, file=sys.stderr)
    sys.exit(1)

try:
    import librosa
except Exception as e:
    print("[FATAL] librosa is required for resampling/mono conversion:", e, file=sys.stderr)
    sys.exit(1)

# --- DataFrames ---
try:
    import pandas as pd
except Exception as e:
    print("[FATAL] pandas is required for tabular outputs:", e, file=sys.stderr)
    sys.exit(1)


# ---- Helpers ----------------------------------------------------------------

TARGET_SR = 16_000

def find_wavs(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])

def normalize_to_16k_mono(src: Path, dst: Path) -> None:
    """
    Load audio with librosa, force mono + 16 kHz, write PCM_16 WAV with soundfile.
    """
    y, sr = librosa.load(str(src), sr=TARGET_SR, mono=True)
    # Guard against completely silent / very short signals
    if y is None or y.size == 0:
        y = np.zeros(TARGET_SR // 10, dtype=np.float32)  # 0.1s of silence
    sf.write(str(dst), y, TARGET_SR, subtype="PCM_16")

def normalize_tree(input_dir: Path) -> Path:
    """
    Create a temp tree mirroring input_dir, with every .wav converted to 16k mono.
    Returns the root of the normalized tree.
    """
    tmp_root = Path(tempfile.mkdtemp(prefix="disvoice_norm_"))
    for wav in find_wavs(input_dir):
        rel = wav.relative_to(input_dir)
        out_path = (tmp_root / rel).with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            normalize_to_16k_mono(wav, out_path)
        except Exception as e:
            print(f"[WARN] Failed to normalize {wav}: {e}", file=sys.stderr)
    return tmp_root

def run_family_per_file(
    spec: "FamilySpec",
    input_dir: Path,
    static: bool = True,
    plots: bool = False,
    fmt: str = "dataframe",
) -> pd.DataFrame:
    """Robust path-join–free extraction: call extract_features_file for each wav."""
    Cls = _lazy_import(spec.import_path, spec.class_name)
    extractor = Cls(**spec.kwargs) if spec.kwargs else Cls()

    if not hasattr(extractor, "extract_features_file"):
        raise AttributeError(f"{spec.class_name} has no extract_features_file method")

    rows = []
    wavs = find_wavs(input_dir)
    if not wavs:
        return pd.DataFrame()

    for wav in wavs:
        try:
            df = extractor.extract_features_file(
                str(wav), static=static, plots=plots, fmt=fmt
            )
            if not isinstance(df, pd.DataFrame):
                raise RuntimeError(f"Unexpected return type: {type(df)} for {wav}")
            # Ensure filename column exists and is just the basename
            if "filename" not in df.columns:
                df.insert(0, "filename", wav.name)
            rows.append(df)
        except Exception as e:
            print(f"[WARN] {spec.class_name} failed on {wav.name}: {e}", file=sys.stderr)

    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()



# ---- Family runners ----------------------------------------------------------

@dataclass
class FamilySpec:
    import_path: str                 # e.g., "disvoice.glottal"
    class_name: str                  # e.g., "Glottal"
    kwargs: Dict[str, object]        # extra kwargs for constructor or call
    needs_pkgs: tuple[str, ...] = () # optional: soft deps to warn about

def _lazy_import(module: str, symbol: str):
    import importlib
    try:
        mod = importlib.import_module(module)
        return getattr(mod, symbol)
    except Exception as e:
        raise ImportError(f"Cannot import {symbol} from {module}: {e}")

def run_family(
    spec: "FamilySpec",
    input_dir: Path,
    static: bool = True,
    plots: bool = False,
    fmt: str = "dataframe",
) -> Optional[pd.DataFrame]:
    # Soft-dependency checks (warn only)
    for pkg in spec.needs_pkgs:
        try:
            __import__(pkg)
        except Exception:
            print(f"[WARN] Optional dependency '{pkg}' not available; "
                  f"{spec.class_name} may fail or be incomplete.", file=sys.stderr)

    # Prefer robust per-file extraction when available
    try:
        return run_family_per_file(spec, input_dir, static=static, plots=plots, fmt=fmt)
    except AttributeError:
        pass  # no per-file API, try path API below

    # Fallback: path-level API
    Cls = _lazy_import(spec.import_path, spec.class_name)
    extractor = Cls(**spec.kwargs) if spec.kwargs else Cls()

    input_base = str(input_dir)
    # Adding a separator won’t always survive abspath(), but keep it anyway.
    if not input_base.endswith(os.sep):
        input_base += os.sep

    df = extractor.extract_features_path(
        input_base,
        static=static,
        plots=plots,
        fmt=fmt,
    )
    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"{spec.class_name} returned non-DataFrame type: {type(df)}")
    return df



FAMILIES: Dict[str, FamilySpec] = {
    "glottal": FamilySpec("disvoice.glottal", "Glottal", {}, needs_pkgs=()),
    "phonation": FamilySpec("disvoice.phonation", "Phonation", {}, needs_pkgs=("parselmouth",)),  # Praat backend
    "articulation": FamilySpec("disvoice.articulation", "Articulation", {}, needs_pkgs=()),
    "prosody": FamilySpec("disvoice.prosody", "Prosody", {}, needs_pkgs=()),
    "phonological": FamilySpec("disvoice.phonological", "Phonological", {}, needs_pkgs=("phonet",)),
    "replearning": FamilySpec("disvoice.replearning", "RepLearning", {"model": "CAE"}, needs_pkgs=("tensorflow",)),
}


# ---- Main orchestration ------------------------------------------------------

def save_table(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"out_{name}.csv"
    json_path = out_dir / f"out_{name}.json"

    # Consistent column order: ensure filename (if present) is first
    cols = list(df.columns)
    if "filename" in cols:
        cols = ["filename"] + [c for c in cols if c != "filename"]
        df = df[cols]

    df.to_csv(csv_path, index=False)
    try:
        # pandas >= 2.0 supports indent
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)
    except TypeError:
        # Fallback if indent not supported
        df.to_json(json_path, orient="records", force_ascii=False)

    print(f"[OK] {name:<12} → {csv_path.name} / {json_path.name} "
          f"({len(df)} rows, {len(df.columns)} cols)")

def main():
    ap = argparse.ArgumentParser(description="Extract all DisVoice feature families to CSV and JSON.")
    ap.add_argument("input", type=Path, help="Folder with WAV files (recursively scanned).")
    ap.add_argument("-o", "--out", type=Path, default=Path("./features_out"), help="Output folder.")
    ap.add_argument("--include", nargs="*", default=list(FAMILIES.keys()),
                    help=f"Subset of families to run (default: all). Choices: {', '.join(FAMILIES.keys())}")
    ap.add_argument("--model", choices=["CAE", "RAE"], default="CAE",
                    help="RepLearning model to use (default: CAE).")
    ap.add_argument("--keep-temp", action="store_true", help="Do not delete normalized temp audio.")
    args = ap.parse_args()

    input_dir: Path = args.input
    out_dir: Path = args.out
    include = [f.lower() for f in args.include]

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[FATAL] Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(2)

    # Ensure DisVoice is importable when running inside the repo
    try:
        import disvoice  # noqa: F401
    except Exception:
        repo_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_root))
        try:
            import disvoice  # noqa: F401
        except Exception as e:
            print("[FATAL] Cannot import 'disvoice'. Install it (pip -e .) or run from repo root.", e, file=sys.stderr)
            sys.exit(3)

    # Patch RepLearning model choice
    if "replearning" in include:
        FAMILIES["replearning"].kwargs["model"] = args.model

    print("[INFO] Normalizing all WAVs to 16 kHz mono…")
    norm_root = normalize_tree(input_dir)
    print(f"[INFO] Normalized tree at: {norm_root}")

    results: Dict[str, pd.DataFrame] = {}
    for fam in include:
        if fam not in FAMILIES:
            print(f"[WARN] Unknown family '{fam}', skipping.", file=sys.stderr)
            continue

        spec = FAMILIES[fam]
        print(f"[INFO] Running {fam}…")
        try:
            df = run_family(spec, norm_root, static=True, plots=False, fmt="dataframe")
            # Attach family label if absent (helps downstream merges)
            if "family" not in df.columns:
                df.insert(0, "family", fam)
            results[fam] = df
            save_table(df, out_dir, fam)
        except Exception as e:
            print(f"[ERROR] {fam} failed: {e}", file=sys.stderr)

    # Optionally, also write a combined table (one row per file per family)
    if results:
        try:
            combined = pd.concat(results.values(), axis=0, ignore_index=True, sort=False)
            save_table(combined, out_dir, "ALL_FAMILIES")
        except Exception as e:
            print(f"[WARN] Could not write combined table: {e}", file=sys.stderr)

    if not args.keep_temp:
        try:
            shutil.rmtree(norm_root, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
