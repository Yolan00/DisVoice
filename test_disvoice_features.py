#!/usr/bin/env python3
"""
DisVoice feature-extraction smoke test for Python 3.13.9

What it does
------------
- Imports the DisVoice package from your working copy (or site-packages if installed).
- Discovers and tests the main extractors (Articulation, Glottal, Phonation, Phonological, Prosody).
- For each extractor:
  * tries to instantiate the class (with no args) and to run a best-guess extraction method
    on a short WAV file (you can pass your own audio, otherwise a synthetic test tone is generated).
  * catches and logs exceptions with a compact traceback.
- Prints a Pass/Fail summary and writes a JSON report.

How to run
----------
    # From the repo root (recommended)
    python test_disvoice_features.py --input path/to/audio.wav --out ./_disvoice_test_out

    # If you don't have a WAV at hand, omit --input; a synthetic 1s 16kHz tone is generated.
    python test_disvoice_features.py

    # More verbosity
    python test_disvoice_features.py -v

Outputs
-------
- Console summary (per extractor)
- JSON report at: <out_dir>/disvoice_test_report.json
"""

from __future__ import annotations
import argparse
import importlib
import inspect
import io
import json
import os
import sys
import traceback
import wave
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------- Utilities ----------

def ensure_repo_on_path(repo_root_hint: Optional[str] = None) -> None:
    """
    Ensure the current repo (if running from a working copy) is importable.
    If `disvoice` is already importable, we do nothing.
    Otherwise we add the provided hint or the cwd to sys.path.
    """
    try:
        importlib.import_module("disvoice")
        return
    except Exception:
        pass

    candidate = repo_root_hint or os.getcwd()
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

@dataclass
class ExtractorSpec:
    name: str
    module_path: str   # e.g., "disvoice.articulation.articulation"
    class_name: str    # e.g., "Articulation"

# Known main extractors commonly present in DisVoice
DEFAULT_EXTRACTORS: List[ExtractorSpec] = [
    ExtractorSpec("Articulation", "disvoice.articulation.articulation", "Articulation"),
    ExtractorSpec("Glottal",      "disvoice.glottal.glottal",          "Glottal"),
    ExtractorSpec("Phonation",    "disvoice.phonation.phonation",      "Phonation"),
    ExtractorSpec("Phonological", "disvoice.phonological.phonological","Phonological"),
    ExtractorSpec("Prosody",      "disvoice.prosody.prosody",          "Prosody"),
    # If your branch includes additional modules, add them here.
    # ExtractorSpec("Relearning", "disvoice.relearning.relearning", "Relearning"),
]

@dataclass
class TestResult:
    extractor: str
    imported: bool
    instantiated: bool
    ran_extraction: bool
    error: Optional[str] = None
    duration_s: float = 0.0
    output_paths: List[str] = None

# ---------- Audio helper ----------

def generate_test_wav(path: str, duration_s: float = 1.0, sr: int = 16000) -> None:
    """
    Generate a simple 1-second 16-bit PCM mono tone with a slow amplitude modulation.
    This is NOT speech, but is enough to exercise I/O and basic DSP paths.
    """
    num_samples = int(duration_s * sr)
    tone_hz = 220.0
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        for n in range(num_samples):
            t = n / sr
            # A soft AM to avoid a perfectly steady tone
            amp = 0.3 * (0.6 + 0.4 * math.sin(2 * math.pi * 2.0 * t))
            sample = int(amp * math.sin(2 * math.pi * tone_hz * t) * 32767.0)
            wf.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))

# ---------- Core test logic ----------

def try_instantiate(cls: type) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Try to instantiate the extractor class with no args first, then with permissive kwargs if needed.
    We avoid guessing complex constructor signatures; we aim for a smoke test only.
    """
    try:
        instance = cls()  # most of DisVoice extractors allow parameterless init
        return True, instance, None
    except Exception as e:
        # If constructor requires args, report and skip. We don't brute-force unknown signatures.
        return False, None, f"Instantiation failed: {type(e).__name__}: {e}"

def candidate_methods(obj: Any) -> List[str]:
    """
    Return a prioritized list of method names to try for 'run a single-file extraction'.
    Adjust the list if your branch uses different method names.
    """
    return [
        "extract_features_file",
        "extract_features_path",
        # fallbacks, just in case
        "extract_features",
        "extract",
        "run",
    ]

def call_best_method(instance: Any, audio_path: str, out_dir: str) -> Tuple[bool, List[str], Optional[str]]:
    """
    Try to call a plausible extraction method with minimal arguments.
    We try multiple common signatures:
        method(audio_path)
        method(audio_path, out_dir=...)
        method([audio_path], out_dir=...)
    We collect any produced file paths if the method returns them (list/str) or prints them.
    """
    outputs: List[str] = []
    for m in candidate_methods(instance):
        if not hasattr(instance, m) or not callable(getattr(instance, m)):
            continue
        fn = getattr(instance, m)

        # Try several safe call patterns
        tried_signatures = [
            # single file, with optional out_dir
            ((audio_path,), {"out_dir": out_dir}),
            ((audio_path,), {}),
            # some modules accept 'path'/'filename' kw names
            ((), {"filename": audio_path, "out_dir": out_dir}),
            ((), {"path": audio_path, "out_dir": out_dir}),
            # path list variants
            (([audio_path],), {"out_dir": out_dir}),
            (([audio_path],), {}),
        ]

        for args, kwargs in tried_signatures:
            try:
                ret = fn(*args, **kwargs)
                # Collect outputs if any
                if isinstance(ret, str):
                    outputs.append(ret)
                elif isinstance(ret, (list, tuple)):
                    outputs.extend([str(x) for x in ret])
                return True, outputs, None
            except TypeError as te:
                # Signature mismatch; try next pattern
                last_sig_err = te
                continue
            except Exception as e:
                # Real runtime error -> report immediately for this method
                tb = traceback.format_exc(limit=6)
                return False, outputs, f"{type(e).__name__}: {e}\n{tb}"

    return False, outputs, "No suitable extraction method found on the instance."

def test_extractor(spec: ExtractorSpec, audio_path: str, out_dir: str, verbose: bool=False) -> TestResult:
    t0 = time.time()
    result = TestResult(extractor=spec.name, imported=False, instantiated=False,
                        ran_extraction=False, error=None, duration_s=0.0, output_paths=[])

    try:
        mod = importlib.import_module(spec.module_path)
        result.imported = True
        if verbose:
            print(f"[Import OK] {spec.module_path}")
    except Exception as e:
        result.error = f"Import failed: {type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"
        result.duration_s = time.time() - t0
        return result

    try:
        cls = getattr(mod, spec.class_name)
    except AttributeError:
        result.error = f"Class '{spec.class_name}' not found in module '{spec.module_path}'."
        result.duration_s = time.time() - t0
        return result

    ok_inst, instance, err = try_instantiate(cls)
    result.instantiated = ok_inst
    if not ok_inst:
        result.error = err
        result.duration_s = time.time() - t0
        return result
    if verbose:
        print(f"[New {spec.class_name}] instantiated")

    ok_run, outs, err = call_best_method(instance, audio_path, out_dir)
    result.ran_extraction = ok_run
    result.output_paths = outs
    if not ok_run:
        result.error = err
    result.duration_s = time.time() - t0
    return result

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Smoke test DisVoice feature extractors.")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to a WAV file to test. If omitted, a synthetic test file is generated.")
    parser.add_argument("--out", "-o", type=str, default="./_disvoice_test_out",
                        help="Directory for temporary outputs and the JSON report.")
    parser.add_argument("--repo-root", type=str, default=None,
                        help="(Optional) Path to your repo root; added to sys.path if disvoice is not importable.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    ensure_repo_on_path(args.repo_root)

    # Confirm we can import the top-level package (and show version if available)
    try:
        disvoice_pkg = importlib.import_module("disvoice")
        top_level_ok = True
    except Exception as e:
        top_level_ok = False
        print("[FATAL] Could not import 'disvoice' package. Are you running from the repo or is it installed?",
              file=sys.stderr)
        print(type(e).__name__, ":", e, file=sys.stderr)
        sys.exit(1)

    version = getattr(disvoice_pkg, "__version__", "unknown")
    print(f"== DisVoice import OK (version: {version}) ==")

    os.makedirs(args.out, exist_ok=True)

    # Prepare an audio file
    if args.input is None:
        gen_path = os.path.join(args.out, "test_tone_16k.wav")
        generate_test_wav(gen_path, duration_s=1.0, sr=16000)
        audio_path = gen_path
        if args.verbose:
            print(f"[Audio] Generated synthetic WAV at {audio_path}")
    else:
        audio_path = os.path.abspath(args.input)
        if not os.path.isfile(audio_path):
            print(f"[FATAL] Input file not found: {audio_path}", file=sys.stderr)
            sys.exit(2)

    # Run tests
    results: List[TestResult] = []
    print("\n== Running extractor smoke tests ==")
    for spec in DEFAULT_EXTRACTORS:
        print(f"\n-- {spec.name} --")
        res = test_extractor(spec, audio_path, args.out, verbose=args.verbose)
        results.append(res)
        status = "PASS" if (res.imported and res.instantiated and res.ran_extraction and res.error is None) else "FAIL"
        print(f"  Import:       {'OK' if res.imported else 'FAIL'}")
        print(f"  Instantiate:  {'OK' if res.instantiated else 'FAIL'}")
        print(f"  Run method:   {'OK' if res.ran_extraction else 'FAIL'}")
        print(f"  Duration:     {res.duration_s:.2f}s")
        if res.output_paths:
            print(f"  Outputs:      {len(res.output_paths)} (first: {res.output_paths[0]})")
        if res.error:
            print("  Error:")
            print("  " + "\n  ".join(res.error.strip().splitlines()))
        print(f"  ==> {status}")

    # Summary
    passed = sum(1 for r in results if r.imported and r.instantiated and r.ran_extraction and not r.error)
    total = len(results)
    print("\n== Summary ==")
    print(f"Passed {passed}/{total} extractors.")

    # Write JSON report
    report_path = os.path.join(args.out, "disvoice_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Report written to: {report_path}")

    # Exit code reflects success/failure (useful for CI)
    sys.exit(0 if passed == total else 3)

if __name__ == "__main__":
    main()
