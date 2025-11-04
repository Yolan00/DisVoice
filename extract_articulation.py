#!/usr/bin/env python3
"""
Extract all DisVoice feature families from one WAV, save CSVs, and print previews.

- Prosody
- Phonation  (requires: praat executable + praat-parselmouth Python package)
- Articulation
- Glottal    (if you see 'np.Inf' error, change to 'np.inf' in DisVoice source)
- Phonological (may require Keras compile() arg update as discussed)

For each family:
- Uses fmt="dataframe" to get a pandas.DataFrame
- Saves to ./features_out/out_<family>.csv
- Prints shape and head() preview
"""

from pathlib import Path
import os
import sys
import pandas as pd

# ---------------- User paths ----------------
PROJECT = Path("/home/yolan00/Desktop/AND/DisVoice").resolve()
AUDIO   = Path("/home/yolan00/Desktop/AND/data/test.wav").resolve()
OUTDIR  = PROJECT / "features_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Optional: silence Kaldi banner noise
os.environ.setdefault("KALDI_ROOT", "")

# Ensure local editable package is importable
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# ---------------- Imports ----------------
from disvoice.prosody import Prosody
from disvoice.phonation import Phonation
from disvoice.articulation import Articulation
from disvoice.glottal import Glottal
from disvoice.phonological import Phonological

# -------------- Utilities --------------
def ensure_audio(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}\n"
                                f"Put a mono 16 kHz WAV in AND/data/ and update AUDIO.")
    return str(path)

def run_family(name: str, obj, wav_path: str, out_csv: Path,
               static: bool = True, plots: bool = False, fmt: str = "dataframe"):
    """
    Run one feature family using the documented extract_features_file API.
    - Returns the DataFrame on success, or None on failure (after printing error).
    """
    print(f"\n=== {name.upper()} ===")
    try:
        df = obj.extract_features_file(wav_path, static=static, plots=plots, fmt=fmt)
        if df is None:
            print(f"[WARN] {name}: returned None with fmt='{fmt}'. "
                  f"Try fmt='npy' to inspect raw output or verify the audio.")
            return None

        # Save CSV & print preview
        df.to_csv(out_csv, index=False)
        print(f"[OK] {name} → {out_csv}")
        print(f"[INFO] {name} shape: {getattr(df, 'shape', None)}")
        # Print a short, readable preview (up to 5 rows, 120 chars wide)
        with pd.option_context("display.max_rows", 5, "display.max_columns", 10, "display.width", 120):
            print(df.head())

        return df
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        return None

def main():
    wav = ensure_audio(AUDIO)

    # Prosody
    prosody_df = run_family(
        name="prosody",
        obj=Prosody(),
        wav_path=wav,
        out_csv=OUTDIR / "out_prosody.csv",
    )

    # Phonation (Praat + Parselmouth needed)
    phonation_df = run_family(
        name="phonation",
        obj=Phonation(),
        wav_path=wav,
        out_csv=OUTDIR / "out_phonation.csv",
    )

    # Articulation
    articulation_df = run_family(
        name="articulation",
        obj=Articulation(),
        wav_path=wav,
        out_csv=OUTDIR / "out_articulation.csv",
    )

    # Glottal (NumPy 2.0: ensure DisVoice source uses np.inf, not np.Inf)
    glottal_df = run_family(
        name="glottal",
        obj=Glottal(),
        wav_path=wav,
        out_csv=OUTDIR / "out_glottal.csv",
    )

    # Phonological (Keras changes may be needed; errors will be printed)
    phonological_df = run_family(
        name="phonological",
        obj=Phonological(),
        wav_path=wav,
        out_csv=OUTDIR / "out_phonological.csv",
    )

    # Summary
    print("\n=== SUMMARY ===")
    for n, df in {
        "prosody": prosody_df,
        "phonation": phonation_df,
        "articulation": articulation_df,
        "glottal": glottal_df,
        "phonological": phonological_df,
    }.items():
        print(f"{n:13s}: {'OK' if df is not None else 'FAILED'}")

    print(f"\nOutputs are in: {OUTDIR}")

if __name__ == "__main__":
    main()
