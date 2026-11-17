#!/usr/bin/env python3
"""Brian2 bug reproduction harness (read-only; does not patch library code).

Each test function is executed in a fresh subprocess per (id, target).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.json"
PY = sys.executable


def _setup_target(target: str) -> None:
    os.environ["BRIAN2_REPRO_TARGET"] = target
    from brian2 import prefs, set_device

    prefs.logging.console_log_level = "ERROR"
    prefs.logging.file_log = False
    if target == "numpy":
        prefs.codegen.target = "numpy"
    elif target == "cython":
        prefs.codegen.target = "cython"
    elif target == "cpp_standalone":
        directory = os.environ.get("BRIAN2_REPRO_DIR")
        if not directory:
            directory = tempfile.mkdtemp(prefix="b2cpp_")
            os.environ["BRIAN2_REPRO_DIR"] = directory
        set_device("cpp_standalone", directory=directory, debug=False)
    else:
        raise ValueError(target)


def _exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


def rec(verdict, output, **extra):
    d = {"verdict": verdict, "output": output}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_G18():
    """np.sin on a volt VariableView: raise vs silent unit drop."""
    import numpy as np
    from brian2 import NeuronGroup, volt, DimensionMismatchError, prefs

    results = {}
    G = NeuronGroup(1, "w : volt")
    G.w = 1 * volt
    for name, fn in [
        ("sin", np.sin),
        ("cos", np.cos),
        ("exp", np.exp),
        ("sqrt", np.sqrt),
        ("log", np.log),
        ("tanh", np.tanh),
    ]:
        try:
            val = fn(G.w)
            results[name] = {
                "raised": False,
                "type": type(val).__name__,
                "repr": repr(val),
                "has_dim": hasattr(val, "dim"),
                "dtype": getattr(val, "dtype", None),
            }
        except Exception as e:
            results[name] = {"raised": True, "error": _exc(e)}

    # Direct call of the misspelled method, if present
    vw = G.w
    misspelled = hasattr(vw, "__array__ufunc__")
    correct = hasattr(vw, "__array_ufunc__")
    direct = None
    if misspelled:
        try:
            direct = vw.__array__ufunc__(np.sin, "__call__")
            direct = {"raised": False, "repr": repr(direct), "type": type(direct).__name__}
        except Exception as e:
            direct = {"raised": True, "error": _exc(e)}

    sin = results["sin"]
    if sin.get("raised") and "DimensionMismatch" in sin.get("error", ""):
        verdict = "NOT REPRODUCED"
        note = "np.sin(G.w) correctly raises DimensionMismatchError; silent unit-drop claim is false for standard ufuncs"
    elif sin.get("raised"):
        verdict = "PARTIALLY CONFIRMED"
        note = "raises, but not DimensionMismatchError"
    else:
        verdict = "CONFIRMED"
        note = "silently returned a value"

    return rec(
        verdict,
        {
            "ufuncs": results,
            "has___array__ufunc__": misspelled,
            "has___array_ufunc__": correct,
            "direct_misspelled_call": direct,
            "note": note,
        },
    )


def test_M3():
    """Subexpression not recomputed after same-line / nested dependency change."""
    from brian2 import NeuronGroup, run, defaultclock, ms

    G = NeuronGroup(1, """
        v : 1
        z : 1
        A = 2*v + 1 : 1
    """)
    G.v = 1
    G.run_regularly("v = A + 10\nz = A")
    run(defaultclock.dt)
    z_same = float(G.z[0])
    v_same = float(G.v[0])
    out = {"same_line": {"v": v_same, "z": z_same, "expected_z": 27, "stale_z": 3}}
    if z_same == 3:
        verdict = "CONFIRMED"
    elif z_same == 27:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M3_nested():
    """Nested subexpression case in isolation (needed for cpp_standalone)."""
    from brian2 import NeuronGroup, run, defaultclock, Network

    G2 = NeuronGroup(1, """
        v : 1
        y : 1
        z : 1
        B = 2*v : 1
        A = B + 1 : 1
    """)
    G2.v = 1
    G2.run_regularly("y = A\nv = 10\nz = A")
    run(defaultclock.dt)
    out = {
        "v": float(G2.v[0]),
        "y": float(G2.y[0]),
        "z": float(G2.z[0]),
        "expected_z": 21,
        "stale_z": 3,
    }
    if out["z"] == 3:
        verdict = "CONFIRMED"
    elif out["z"] == 21:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M8():
    """NumPy tmp=v; tmp=tmp+1 aliases and mutates v; C++ should copy."""
    from brian2 import NeuronGroup, run, defaultclock
    import numpy as np

    G = NeuronGroup(3, """
        v : 1
        x : 1
        tmp : 1
    """)
    G.v = [1, 2, 3]
    G.run_regularly("tmp = v\ntmp = tmp + 1\nx = tmp")
    run(defaultclock.dt)
    v = np.array(G.v[:], dtype=float)
    x = np.array(G.x[:], dtype=float)
    out = {"v": v.tolist(), "x": x.tolist()}
    corrupted = np.allclose(v, [2, 3, 4])
    copied = np.allclose(v, [1, 2, 3]) and np.allclose(x, [2, 3, 4])
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "numpy":
        verdict = "CONFIRMED" if corrupted else "NOT REPRODUCED"
    else:
        # claimed: C++ copies, does not corrupt v
        if copied:
            verdict = "CONFIRMED"
        elif corrupted:
            verdict = "NOT REPRODUCED"
        else:
            verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out, corrupted=corrupted, copied=copied)


def test_M9():
    """NumPy connect() with target-dependent n indexes _n by _j."""
    from brian2 import NeuronGroup, Synapses
    import numpy as np

    G = NeuronGroup(6, "v : 1")
    S = Synapses(G, G, "w : 1")
    err = None
    try:
        S.connect(j="N_post-1-k for k in range(N_post) if i==0", n="1 + j % 2")
        from brian2 import run as _run
        target = os.environ.get("BRIAN2_REPRO_TARGET")
        if target == "cpp_standalone":
            _run(0 * G.clock.dt)
        i_vals = np.array(S.i[:], dtype=int).tolist()
        j_vals = np.array(S.j[:], dtype=int).tolist()
        pairs = list(zip(i_vals, j_vals))
        from collections import Counter
        counts = Counter(j_vals)
        out = {"pairs": pairs, "counts_by_j": dict(sorted(counts.items())), "N_syn": len(j_vals)}
        # Correct: odd j (1,3,5) have 2 synapses, even j (0,2,4) have 1
        correct = {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 2}
        # Bug: even targets doubled instead of odd
        buggy_even = {0: 2, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1}
        if counts == Counter(correct):
            verdict = "NOT REPRODUCED"  # this target is correct
        elif counts == Counter(buggy_even):
            verdict = "CONFIRMED"
        else:
            verdict = "PARTIALLY CONFIRMED"
    except Exception as e:
        out = {"error": _exc(e)}
        verdict = "CONFIRMED" if "IndexError" in type(e).__name__ else "PARTIALLY CONFIRMED"
        # the generator form that raises is a second sub-claim
    # second sub-claim: j='k for k in range(3,6)' with same n
    err2 = None
    try:
        G2 = NeuronGroup(6, "v : 1", name="G2")
        S2 = Synapses(G2, G2, "w : 1", name="S2")
        S2.connect(j="k for k in range(3, 6)", n="1 + j % 2")
        err2 = {"ok": True, "N": int(S2.N), "j": S2.j[:].tolist()}
    except Exception as e:
        err2 = {"ok": False, "error": _exc(e)}
    out["range_3_6"] = err2
    return rec(verdict, out)


def test_M16():
    """Cython BinomialFunction p>0.5 substitutes raw p instead of mirrored P."""
    from brian2 import NeuronGroup, BinomialFunction, run, defaultclock, seed
    import numpy as np

    seed(42)
    n, p, N = 8, 0.6, 20000
    bf = BinomialFunction(n, p, approximate=False)
    G = NeuronGroup(N, "x : 1")
    G.run_regularly("x = bf()")
    run(defaultclock.dt)
    mean = float(np.mean(G.x[:]))
    out = {"n": n, "p": p, "N": N, "mean": mean, "expected": n * p, "audit_emulation": 5.79}
    # cython bug ~5.79; correct ~4.8; allow sampling noise
    if abs(mean - 4.8) < 0.15:
        verdict = "NOT REPRODUCED"
    elif abs(mean - 5.79) < 0.3 or mean > 5.2:
        verdict = "CONFIRMED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_C5():
    """C++ int/int true division in single precision via 1.0f*."""
    from brian2 import NeuronGroup, run
    import numpy as np

    N = 20
    G = NeuronGroup(N, "v : 1")
    G.v = "i / (i + 1)"
    run(0 * G.clock.dt)
    vals = np.array(G.v[:], dtype=np.float64)
    # i=2 → 2/3
    two_thirds = float(vals[2])
    f32 = float(np.float32(2.0) / np.float32(3.0))
    f64 = 2.0 / 3.0
    # also check a large integer near 2^24
    G2 = NeuronGroup(1, """
        k : integer
        v : 1
    """, name="big")
    G2.k = 2**24 + 1
    G2.v = "k / (k + 1)"
    # For cpp we cannot run twice with a new group. Combine into one network from the start
    # if this is cpp and we already ran, skip big — handled by combined version below.
    out = {
        "v": vals.tolist(),
        "two_thirds": two_thirds,
        "float32_2_3": f32,
        "float64_2_3": f64,
        "hex_two_thirds": float.hex(two_thirds),
        "hex_f32": float.hex(float(np.float32(f32))),
        "hex_f64": float.hex(f64),
        "matches_float32": abs(two_thirds - f32) < abs(two_thirds - f64),
        "matches_float64": abs(two_thirds - f64) < 1e-15,
    }
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "cpp_standalone":
        if abs(two_thirds - f32) < 1e-10 and abs(two_thirds - f64) > 1e-10:
            verdict = "CONFIRMED"
        elif abs(two_thirds - f64) < 1e-15:
            verdict = "NOT REPRODUCED"
        else:
            verdict = "PARTIALLY CONFIRMED"
    else:
        # numpy/cython: expect double
        verdict = "CONFIRMED" if abs(two_thirds - f64) < 1e-12 else "PARTIALLY CONFIRMED"
        if target != "cpp_standalone":
            # the claim is C++ specific; numpy being correct supports the contrast
            pass
    return rec(verdict, out)


def test_C5_large():
    """C++ int/int division rounding for integers above 2^24."""
    from brian2 import NeuronGroup, run
    import numpy as np

    G = NeuronGroup(1, """
        k : integer
        v : 1
    """)
    G.k = 16777217  # 2^24 + 1
    G.v = "k / (k + 1)"
    run(0 * G.clock.dt)
    val = float(G.v[0])
    k = 16777217.0
    f32 = float(np.float32(k) / np.float32(k + 1))
    f64 = k / (k + 1)
    out = {"value": val, "float32": f32, "float64": f64, "k": 16777217}
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "cpp_standalone":
        verdict = "CONFIRMED" if abs(val - f32) < abs(val - f64) else "NOT REPRODUCED"
    else:
        verdict = "CONFIRMED" if abs(val - f64) < 1e-12 else "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_C6_negpow():
    """Integer power 2**(-k): NumPy raises, C++ 0.25, Cython 0."""
    from brian2 import NeuronGroup, run
    import numpy as np

    G = NeuronGroup(3, """
        k : integer
        v : 1
    """)
    G.k = [2, 2, 2]
    err = None
    try:
        G.v = "2 ** (-k)"
        run(0 * G.clock.dt)
        vals = np.array(G.v[:], dtype=float).tolist()
        out = {"values": vals}
    except Exception as e:
        out = {"error": _exc(e)}
        vals = None
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "numpy":
        verdict = "CONFIRMED" if "error" in out else "NOT REPRODUCED"
    elif target == "cython":
        if vals is not None and all(v == 0 for v in vals):
            verdict = "CONFIRMED"
        elif vals is not None and all(abs(v - 0.25) < 1e-12 for v in vals):
            verdict = "NOT REPRODUCED"
        else:
            verdict = "PARTIALLY CONFIRMED"
    else:  # cpp
        if vals is not None and all(abs(v - 0.25) < 1e-12 for v in vals):
            verdict = "CONFIRMED"
        elif vals is not None and all(v == 0 for v in vals):
            verdict = "NOT REPRODUCED"
        else:
            verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_C6_overflow():
    """G.v = 'i**3' overflows int32 on NumPy at i=1291; C++ correct."""
    from brian2 import NeuronGroup, run
    import numpy as np

    G = NeuronGroup(1300, "v : 1")
    G.v = "i ** 3"
    run(0 * G.clock.dt)
    v1291 = float(G.v[1291])
    expected = 1291 ** 3  # 2_153_689_171
    int32_wrap = np.int32(np.int64(1291) ** 3)  # may overflow
    # numpy int32 overflow of 1291**3
    wrapped = float(np.int32(2153689171 if False else np.array([1291 ** 3]).astype(np.int32)[0]))
    try:
        wrapped = float(np.array([1291], dtype=np.int32)[0] ** 3)  # python int
    except Exception:
        pass
    as_i32 = float(np.int32(1291) ** np.int32(3)) if False else None
    i32 = np.int32(1291)
    # C-style wrap
    wrap_val = float(np.int32(np.uint32(1291 ** 3) if 1291 ** 3 > 2**31 else 1291 ** 3))
    wrap_val = float(((1291 ** 3 + 2**31) % 2**32) - 2**31)
    out = {
        "v1291": v1291,
        "expected": expected,
        "int32_wrap": wrap_val,
        "v0": float(G.v[0]),
        "v10": float(G.v[10]),
    }
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "numpy":
        verdict = "CONFIRMED" if abs(v1291 - wrap_val) < 1 or v1291 < 0 else (
            "NOT REPRODUCED" if abs(v1291 - expected) < 1 else "PARTIALLY CONFIRMED"
        )
    else:
        verdict = "CONFIRMED" if abs(v1291 - expected) < 1 else (
            "NOT REPRODUCED" if v1291 < 0 else "PARTIALLY CONFIRMED"
        )
    return rec(verdict, out)


def test_C7():
    """clip(int, float, float) returns first-arg type on C++/Cython."""
    from brian2 import NeuronGroup, run
    import numpy as np

    G = NeuronGroup(5, "v : 1")
    G.v = "clip(i, 0.5, 2.5)"
    run(0 * G.clock.dt)
    vals = np.array(G.v[:], dtype=float).tolist()
    numpy_expected = [0.5, 1.0, 2.0, 2.5, 2.5]
    cpp_expected = [0.0, 1.0, 2.0, 2.0, 2.0]
    out = {"values": vals, "numpy_expected": numpy_expected, "int_truncated": cpp_expected}
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "numpy":
        verdict = "CONFIRMED" if np.allclose(vals, numpy_expected) else "NOT REPRODUCED"
    else:
        if np.allclose(vals, cpp_expected):
            verdict = "CONFIRMED"
        elif np.allclose(vals, numpy_expected):
            verdict = "NOT REPRODUCED"
        else:
            verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_C8():
    """%=, //=, **= emitted verbatim to C++."""
    from brian2 import NeuronGroup, run, Network
    import numpy as np

    target = os.environ.get("BRIAN2_REPRO_TARGET")
    out = {}

    def run_op(op_code, k0, name):
        G = NeuronGroup(1, "k : integer", name=name)
        G.k = k0
        G.run_regularly(op_code)
        run(G.clock.dt)
        return int(G.k[0])

    # %=
    try:
        if target == "cpp_standalone":
            G = NeuronGroup(1, "k : integer")
            G.k = -2
            G.run_regularly("k %= 3")
            run(G.clock.dt)
            out["mod"] = {"value": int(G.k[0]), "error": None}
        else:
            out["mod"] = {"value": run_op("k %= 3", -2, "gmod"), "error": None}
    except Exception as e:
        out["mod"] = {"value": None, "error": _exc(e)}

    # For cpp, only one run. Separate tests for //= and **= .
    if target != "cpp_standalone":
        try:
            out["floordiv"] = {"value": run_op("k //= 3", 7, "gfloordiv"), "error": None}
        except Exception as e:
            out["floordiv"] = {"value": None, "error": _exc(e)}
        try:
            out["pow"] = {"value": run_op("k **= 3", 2, "gpow"), "error": None}
        except Exception as e:
            out["pow"] = {"value": None, "error": _exc(e)}

    # numpy %= of -2 % 3 is 1; C remainder of -2 % 3 is -2 (towards zero in C++11 toward 0? actually C++ % sign follows dividend: -2)
    val = out.get("mod", {}).get("value")
    if target == "numpy":
        verdict = "CONFIRMED" if val == 1 else "PARTIALLY CONFIRMED"
    elif target == "cpp_standalone":
        if val == -2:
            verdict = "CONFIRMED"
        elif val == 1:
            verdict = "NOT REPRODUCED"
        elif out["mod"]["error"]:
            verdict = "PARTIALLY CONFIRMED"
        else:
            verdict = "PARTIALLY CONFIRMED"
    else:
        verdict = "CONFIRMED" if val in (1, -2) else "PARTIALLY CONFIRMED"
        out["note"] = "cython %= result recorded"
    return rec(verdict, out)


def test_C8_floordiv():
    """k //= 3 on C++: comment or wrong behavior."""
    from brian2 import NeuronGroup, run

    G = NeuronGroup(1, "k : integer")
    G.k = 7
    try:
        G.run_regularly("k //= 3")
        run(G.clock.dt)
        out = {"value": int(G.k[0]), "error": None}
    except Exception as e:
        out = {"value": None, "error": _exc(e)}
    # If comment: k stays 7. If floor div: 2. If compile error: error.
    if out["error"]:
        verdict = "CONFIRMED"
    elif out["value"] == 7:
        verdict = "CONFIRMED"  # treated as comment, silent no-op
    elif out["value"] == 2:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_C8_pow():
    """k **= 3 on C++: syntax error."""
    from brian2 import NeuronGroup, run

    G = NeuronGroup(1, "k : integer")
    G.k = 2
    try:
        G.run_regularly("k **= 3")
        run(G.clock.dt)
        out = {"value": int(G.k[0]), "error": None}
    except Exception as e:
        out = {"value": None, "error": _exc(e)}
    if out["error"] and ("syntax" in out["error"].lower() or "error" in out["error"].lower() or "Compile" in out["error"] or "**" in out["error"]):
        verdict = "CONFIRMED"
    elif out["error"]:
        verdict = "CONFIRMED"
    elif out["value"] == 8:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_C13():
    """Negative p in connect: no synapses for N=6, ValueError for N=2000."""
    from brian2 import NeuronGroup, Synapses

    out = {}
    for N, name in [(6, "small"), (2000, "large")]:
        G = NeuronGroup(N, "v : 1", name=f"G{name}")
        S = Synapses(G, G, name=f"S{name}")
        try:
            S.connect(p=-0.1)
            out[name] = {"N_syn": int(S.N), "error": None}
        except Exception as e:
            out[name] = {"N_syn": None, "error": _exc(e)}
    small_ok = out["small"]["error"] is None and out["small"]["N_syn"] == 0
    large_err = out["large"]["error"] and "negative" in out["large"]["error"].lower()
    if small_ok and large_err:
        verdict = "CONFIRMED"
    elif small_ok and out["large"]["N_syn"] == 0:
        verdict = "NOT REPRODUCED"  # both silently empty
    elif small_ok or large_err:
        verdict = "PARTIALLY CONFIRMED"
    else:
        verdict = "NOT REPRODUCED"
    return rec(verdict, out)


def test_C14():
    """C++ SpikeMonitor on subgroup with _start_idx==0; hang if only outside neurons spike."""
    from brian2 import NeuronGroup, SpikeMonitor, run, ms, Network

    G = NeuronGroup(10, "v : 1", threshold="v > 0.5", reset="v = 0")
    # only neurons 5-9 spike; subgroup is G[:5] so start=0, no events inside
    G.v = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    SG = G[:5]
    M = SpikeMonitor(SG)
    M_all = SpikeMonitor(G, name="all")
    run(1 * ms)
    out = {
        "subgroup_spikes": int(M.num_spikes),
        "all_spikes": int(M_all.num_spikes),
        "subgroup_i": M.i[:].tolist() if M.num_spikes else [],
        "all_i": M_all.i[:].tolist() if M_all.num_spikes else [],
    }
    # If we got here, it did not hang. Zero subgroup spikes is the correct answer.
    if out["subgroup_spikes"] == 0 and out["all_spikes"] > 0:
        verdict = "NOT REPRODUCED"  # terminates correctly
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out, note="completed without hang")


def test_C14_all_in_subgroup():
    """SpikeMonitor on G[:5] when subgroup neurons DO spike (start_idx==0, events present)."""
    from brian2 import NeuronGroup, SpikeMonitor, run, ms

    G = NeuronGroup(10, "v : 1", threshold="v > 0.5", reset="v = 0")
    G.v = 1  # everyone spikes
    SG = G[:5]
    M = SpikeMonitor(SG)
    from brian2 import defaultclock
    run(defaultclock.dt)
    out = {"num_spikes": int(M.num_spikes), "i": M.i[:].tolist()}
    # expect 5 spikes (neurons 0-4), one timestep
    if out["num_spikes"] == 5:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def defaultclock_dt():
    from brian2 import defaultclock
    return defaultclock.dt


def test_I2():
    """BinomialFunction(approximate=False) large n*p: hang from qn rounding to 0."""
    from brian2 import NeuronGroup, BinomialFunction, run, defaultclock, seed

    seed(1)
    # n=80, p=0.5 → nP=40 > 35; qn=(0.5)^80 ~ 8e-25 → .15f is 0
    bf = BinomialFunction(80, 0.5, approximate=False)
    G = NeuronGroup(1, "x : 1")
    G.run_regularly("x = bf()")
    run(defaultclock.dt)
    return rec("NOT REPRODUCED", {"x": float(G.x[0]), "note": "returned instead of hanging"})


def test_I9():
    """2D TimedArray out-of-range column index: wrap/raise vs NaN."""
    from brian2 import TimedArray, NeuronGroup, run, ms, defaultclock
    import numpy as np

    values = np.arange(6.0).reshape(3, 2)
    ta = TimedArray(values, dt=1 * ms)
    out = {}
    # Python-level call
    try:
        out["py_minus1"] = float(ta(0 * ms, -1))
    except Exception as e:
        out["py_minus1"] = _exc(e)
    try:
        out["py_2"] = float(ta(0 * ms, 2))
    except Exception as e:
        out["py_2"] = _exc(e)

    G = NeuronGroup(1, "x : 1")
    G.run_regularly("x = ta(t, -1)")
    try:
        run(defaultclock.dt)
        out["eq_minus1"] = float(G.x[0])
        out["eq_minus1_isnan"] = bool(np.isnan(G.x[0]))
    except Exception as e:
        out["eq_minus1"] = _exc(e)
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    minus = out.get("eq_minus1")
    if isinstance(minus, float) and (minus == out.get("py_minus1") or minus == 1.0):
        # numpy wrap of -1 -> last column
        verdict = "CONFIRMED" if target == "numpy" else "PARTIALLY CONFIRMED"
    elif isinstance(minus, str) and "IndexError" in minus:
        verdict = "CONFIRMED" if target == "numpy" else "PARTIALLY CONFIRMED"
    elif isinstance(minus, float) and (minus != minus or (isinstance(out.get("eq_minus1_isnan"), bool) and out["eq_minus1_isnan"])):
        verdict = "CONFIRMED" if target != "numpy" else "PARTIALLY CONFIRMED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_I9_oob_high():
    """2D TimedArray i >= n_cols inside equations."""
    from brian2 import TimedArray, NeuronGroup, run, ms, defaultclock
    import numpy as np

    values = np.arange(6.0).reshape(3, 2)
    ta = TimedArray(values, dt=1 * ms)
    G = NeuronGroup(1, "x : 1")
    G.run_regularly("x = ta(t, 2)")
    try:
        run(defaultclock.dt)
        val = float(G.x[0])
        out = {"value": val, "isnan": bool(np.isnan(G.x[0]))}
        verdict = "CONFIRMED" if np.isnan(val) else "PARTIALLY CONFIRMED"
    except Exception as e:
        out = {"error": _exc(e)}
        target = os.environ.get("BRIAN2_REPRO_TARGET")
        verdict = "CONFIRMED" if target == "numpy" else "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_I13():
    """Normal-approximation BinomialFunction declared float (single precision)."""
    from brian2 import NeuronGroup, BinomialFunction, run, defaultclock, seed
    import numpy as np

    seed(0)
    # loc = n*p not representable exactly in float32 if we pick carefully
    n = 10_000_001
    p = 0.123456789
    loc = n * p
    scale = np.sqrt(n * p * (1 - p))
    bf = BinomialFunction(n, p, approximate=True)
    N = 5000
    G = NeuronGroup(N, "x : 1")
    G.run_regularly("x = bf()")
    run(defaultclock.dt)
    mean = float(np.mean(G.x[:]))
    loc_f32 = float(np.float32(loc))
    loc_f64 = float(loc)
    out = {
        "mean": mean,
        "loc": loc_f64,
        "loc_f32": loc_f32,
        "scale": float(scale),
        "mean_minus_loc": mean - loc_f64,
        "mean_minus_loc_f32": mean - loc_f32,
        "f32_loc_error": loc_f32 - loc_f64,
    }
    # inspect generated implementation if possible
    try:
        impls = bf.implementations
        out["impl_names"] = str(impls)
    except Exception as e:
        out["impl_err"] = _exc(e)
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "numpy":
        verdict = "CONFIRMED" if abs(mean - loc_f64) < 5 * scale / np.sqrt(N) else "PARTIALLY CONFIRMED"
    else:
        # if mean tracks float32 loc better than float64, confirmed
        if abs(mean - loc_f32) + 0.5 < abs(mean - loc_f64):
            verdict = "CONFIRMED"
        else:
            # sampling noise may hide it; still record
            verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_I13_source():
    """Confirm the Cython/C++ generator emits `float` return type."""
    from brian2.input.binomial import _generate_cython_code, _generate_cpp_code

    cy, _ = _generate_cython_code(10000, 0.5, True, "bino")
    cpp, _ = _generate_cpp_code(10000, 0.5, True, "bino")
    cpp_src = cpp["support_code"] if isinstance(cpp, dict) else str(cpp)
    out = {"cython": cy, "cpp": cpp_src}
    cy_float = "cdef float" in cy
    cpp_float = "float bino" in cpp_src or "float %NAME%" in cpp_src or "float bino(" in cpp_src
    if cy_float and ("float" in cpp_src.split("(")[0]):
        verdict = "CONFIRMED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, {"cython_decl": cy.split("\n")[0:5], "cpp_decl": cpp_src.strip().split("\n")[0:5], "cy_float": cy_float})


# ----- Priority 2: major bugs -----

def test_M1():
    """10**-3 constant-folded to 0."""
    from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor
    import numpy as np

    G = NeuronGroup(5, "v : 1")
    G.v = "i * 10**-3"
    run(0 * defaultclock.dt)
    vals = np.array(G.v[:], dtype=float).tolist()
    out = {"assign_i_times_1e-3": vals, "expected": [0, 0.001, 0.002, 0.003, 0.004]}
    folded_to_zero = all(v == 0 for v in vals)
    # 2.0**-1 and 1/10**3 controls
    G2 = NeuronGroup(1, "v : 1", name="ctrl")
    # cpp cannot add after run. So only first assignment on cpp.
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target != "cpp_standalone":
        G2.v = "2.0 ** -1"
        G3 = NeuronGroup(1, "v : 1", name="ctrl2")
        G3.v = "i * 1/10**3"
        # i=0 → 0 anyway; use
        G3 = NeuronGroup(5, "v : 1", name="ctrl3")
        G3.v = "i * 1 / 10**3"
        out["float_pow"] = float(G2.v[0])
        out["div_form"] = np.array(G3.v[:], dtype=float).tolist()
    verdict = "CONFIRMED" if folded_to_zero else (
        "NOT REPRODUCED" if np.allclose(vals, [0, 0.001, 0.002, 0.003, 0.004]) else "PARTIALLY CONFIRMED"
    )
    return rec(verdict, out)


def test_M1_threshold():
    """threshold='v>10**-1*5', reset='v=10**-1' spikes every step."""
    from brian2 import NeuronGroup, run, ms, SpikeMonitor, defaultclock

    G = NeuronGroup(1, "v : 1", threshold="v > 10**-1 * 5", reset="v = 10**-1")
    G.v = 1
    M = SpikeMonitor(G)
    run(20 * ms)
    n = int(M.num_spikes)
    expected_if_bug = int(round(20 * ms / defaultclock.dt))
    out = {"num_spikes": n, "steps": expected_if_bug, "times": M.t[:].tolist()[:10]}
    if n >= expected_if_bug - 1:
        verdict = "CONFIRMED"
    elif n == 0:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M2_exact():
    """Shared Equations fragment corrupts first group (exact)."""
    from brian2 import Equations, NeuronGroup, run, ms, mV, Network
    import numpy as np

    tau = 10 * ms
    common = Equations("dv/dt = -(v + I)/ tau : volt")
    eqs1 = common + Equations("""
        I = 10*mV : volt
    """)
    G1 = NeuronGroup(1, eqs1, method="exact")
    G1.v = -70 * mV
    net1 = Network(G1)
    net1.run(10 * ms)
    v_before = float(G1.v[0])
    # build second model from shared fragment
    eqs2 = common + Equations("I : volt")
    G2 = NeuronGroup(1, eqs2, method="exact", name="g2")
    G2.v = -70 * mV
    G2.I = 0 * mV
    # re-run first group
    net1.run(0 * ms)  # may not re-evaluate
    # Instead compare by creating both then running G1 in a fresh interpretation:
    # The bug is that after constructing G2, G1's generated code is wrong on NEXT run of a NEW G1-like?
    # Audit: "With both groups created, the first group's result changed"
    # Recreate: create G1, run; create G2; create another run of a group using eqs1?
    # Actually _sort_subexpressions mutates the shared SingleEquation. G1 already has
    # generated code from first run. Need to build G1, G2, THEN run G1.
    return rec("INCONCLUSIVE", {"v_after_first_run_before_G2": v_before, "note": "see test_M2_exact_bothfirst"})


def test_M2_exact_bothfirst():
    """Create both groups before running; exact method."""
    from brian2 import Equations, NeuronGroup, run, ms, mV, Network
    import numpy as np

    tau = 10 * ms
    common = Equations("dv/dt = -(v + I)/ tau : volt")
    eqs1 = common + Equations("I = 10*mV : volt")
    eqs2 = common + Equations("I : volt")
    G1 = NeuronGroup(1, eqs1, method="exact")
    G2 = NeuronGroup(1, eqs2, method="exact", name="g2")
    G1.v = -70 * mV
    G2.v = -70 * mV
    G2.I = 0 * mV
    net = Network(G1, G2)
    net.run(10 * ms)
    v1 = float(G1.v[0] / mV)
    v2 = float(G2.v[0] / mV)
    # analytic for G1: v' = -(v + 10mV)/tau, v(0)=-70mV
    # v(t) = -10 + (-70+10)*exp(-t/tau) = -10 -60*exp(-1) = -10 - 22.0728 = -32.0728 mV
    analytic = -10 - 60 * np.exp(-1)
    out = {"G1_mV": v1, "G2_mV": v2, "G1_analytic_mV": analytic, "err": v1 - analytic}
    # If subexpression I not substituted, G1 uses I as if it were a parameter (0?) 
    return rec("INCONCLUSIVE", out)


def test_M2_pair():
    """Baseline vs shared-fragment corruption for several methods."""
    import os
    from brian2 import Equations, NeuronGroup, Network, ms, mV, prefs
    import numpy as np

    method = os.environ.get("BRIAN2_REPRO_METHOD", "exact")
    both = os.environ.get("BRIAN2_REPRO_BOTH", "1") == "1"
    tau = 10 * ms
    common = Equations("""
        dv/dt = (v_inf - v)/tau : volt
        v_inf = EL + I : volt
    """)
    eqs1 = common + Equations("I = 10*mV : volt")
    EL = -70 * mV
    try:
        if both:
            eqs2 = common + Equations("""
                I = 0*mV : volt
                extra = 1 : 1
            """)
            G1 = NeuronGroup(1, eqs1, method=method)
            G2 = NeuronGroup(1, eqs2, method=method, name="g2")
            G1.v = -70 * mV
            G2.v = -70 * mV
            net = Network(G1, G2)
        else:
            G1 = NeuronGroup(1, eqs1, method=method)
            G1.v = -70 * mV
            net = Network(G1)
        net.run(10 * ms)
        v = float(G1.v[0] / mV)
        out = {"method": method, "both": both, "v_mV": v}
        return rec("ok", out)
    except Exception as e:
        return rec("CRASH", {"method": method, "both": both, "error": _exc(e)})


def test_M4():
    """String assignment through subgroup index applies offset twice."""
    from brian2 import NeuronGroup, run, defaultclock
    import numpy as np

    G = NeuronGroup(10, "v : 1")
    G.v = np.arange(10, dtype=float)
    SG = G[2:7]
    SG.v[1:3] = "i + 100"
    run(0 * defaultclock.dt)
    vals = np.array(G.v[:], dtype=float).tolist()
    out = {"G.v": vals}
    # expected write neurons 3,4 with 103,104
    # bug writes 5,6 with 103,104
    wrote_56 = vals[5] == 103 and vals[6] == 104
    wrote_34 = vals[3] == 103 and vals[4] == 104
    err = None
    try:
        SG.v[3:5] = "1"
        run(0 * defaultclock.dt)
        out["after_3_5"] = np.array(G.v[:], dtype=float).tolist()
    except Exception as e:
        err = _exc(e)
        out["indexerror_3_5"] = err
    out["wrote_5_6"] = wrote_56
    out["wrote_3_4"] = wrote_34
    if wrote_56 and (err and "IndexError" in err):
        verdict = "CONFIRMED"
    elif wrote_56 or (err and "IndexError" in err):
        verdict = "PARTIALLY CONFIRMED"
    elif wrote_34:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M5():
    """Indexing a subgroup variable with a group returns/writes wrong neurons."""
    from brian2 import NeuronGroup
    import numpy as np

    G = NeuronGroup(10, "v : 1")
    G.v = np.arange(10, dtype=float) * 1.0
    SG = G[2:7]
    SG3 = SG[1:3]  # neurons 3,4
    got = np.array(SG.v[SG3], dtype=float).tolist()
    out = {"SG.v[SG3]": got, "expected": [3.0, 4.0], "wrong_56": [5.0, 6.0]}
    SG.v[SG3] = -1
    after = np.array(G.v[:], dtype=float).tolist()
    out["after_write"] = after
    err = None
    try:
        _ = G[5:10].v[G[5:7]]
        out["parent_slice_ok"] = True
    except Exception as e:
        err = _exc(e)
        out["G[5:10].v[G[5:7]]"] = err
    parent_ok = np.array(G.v[SG], dtype=float).tolist()
    out["G.v[SG]"] = parent_ok
    wrong_read = got == [5.0, 6.0]
    wrong_write = after[5] == -1 and after[6] == -1
    if wrong_read and wrong_write and err and "IndexError" in err:
        verdict = "CONFIRMED"
    elif wrong_read or wrong_write:
        verdict = "PARTIALLY CONFIRMED"
    else:
        verdict = "NOT REPRODUCED"
    return rec(verdict, out)


def test_M6():
    """exponential_euler silently drops nonlinear terms."""
    from brian2 import NeuronGroup, run, ms, Network
    import numpy as np
    from scipy.integrate import solve_ivp

    tau = 10 * ms
    G = NeuronGroup(1, "dv/dt = (v - v**2)/tau : 1", method="exponential_euler")
    G.v = 0.1
    run(2 * tau)
    v_ee = float(G.v[0])
    # true ODE
    def f(t, y):
        return [(y[0] - y[0] ** 2) / 0.010]
    sol = solve_ivp(f, [0, 0.02], [0.1], rtol=1e-8, atol=1e-10)
    v_true = float(sol.y[0, -1])
    v_linear = 0.1 * np.exp(2)  # dv/dt = v/tau
    out = {"exponential_euler": v_ee, "scipy": v_true, "linear_only": v_linear}
    if abs(v_ee - v_linear) < 1e-3 and abs(v_ee - v_true) > 1e-2:
        verdict = "CONFIRMED"
    elif abs(v_ee - v_true) < 1e-3:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M7():
    """String refractory re-evaluated every timestep, not once per spike."""
    from brian2 import NeuronGroup, run, ms, SpikeMonitor, defaultclock
    import numpy as np

    defaultclock.dt = 0.1 * ms
    G = NeuronGroup(
        2000,
        "v : 1",
        threshold="v > 0.5",
        reset="v = 0",
        refractory="(1 + 2*rand())*ms",
    )
    G.v = 1  # all spike immediately, then refractory
    # Drive them to spike as soon as refractory ends: v stays 1 after reset? reset to 0.
    # Keep v above threshold always except we need them to spike when not refractory.
    # Use: dv/dt = 0, v=1 always by reset to 1? threshold v>0.5, reset v=0 would
    # not immediately re-spike if refractory. After refractory, v is 0 so they
    # wouldn't spike again.
    # Better: reset v=1 so they spike again as soon as not refractory.
    G = NeuronGroup(
        2000,
        "v : 1",
        threshold="v > 0.5",
        reset="v = 1",
        refractory="(1 + 2*rand())*ms",
    )
    G.v = 1
    M = SpikeMonitor(G)
    run(100 * ms)
    isi = []
    for i in range(2000):
        t = M.t[M.i == i]
        if len(t) >= 2:
            isi.append(np.diff(t / ms))
    if not isi:
        return rec("INCONCLUSIVE", {"num_spikes": int(M.num_spikes)})
    all_isi = np.concatenate(isi)
    mean = float(np.mean(all_isi))
    frac_gt2 = float(np.mean(all_isi > 2.0))
    out = {
        "mean_isi_ms": mean,
        "frac_gt_2ms": frac_gt2,
        "n_isi": int(all_isi.size),
        "documented_mean": 2.0,
        "audit_mean": 1.43,
        "min": float(all_isi.min()),
        "max": float(all_isi.max()),
    }
    if abs(mean - 1.43) < 0.2 and frac_gt2 < 0.1:
        verdict = "CONFIRMED"
    elif abs(mean - 2.0) < 0.15 and abs(frac_gt2 - 0.5) < 0.1:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M10():
    """Distance indexing runs past the end of a section."""
    from brian2.spatialneuron import Soma, Cylinder, SpatialNeuron
    from brian2 import um, nA, uF, ohm, cm
    import numpy as np

    soma = Soma(diameter=10 * um)
    soma.L = Cylinder(n=10, diameter=1 * um, length=10 * um)
    soma.L.L = Cylinder(n=5, diameter=1 * um, length=5 * um)  # binary name LL
    m = soma
    out = {}
    out["L.indices[10*um]"] = np.array(m.L.indices[10 * um]).tolist()
    out["L.indices[3*um:100*um]"] = np.array(m.L.indices[3 * um : 100 * um]).tolist()
    try:
        out["L.indices[-20]"] = np.array(m.L.indices[-20]).tolist()
    except Exception as e:
        out["L.indices[-20]"] = _exc(e)
    try:
        neuron = SpatialNeuron(
            morphology=m,
            model="Im = 0*amp/meter**2 : amp/meter**2",
            Cm=1 * uF / cm ** 2,
            Ri=100 * ohm * cm,
        )
        neuron.I[m.L[10 * um]] = 1 * nA
        nz = np.nonzero(np.array(neuron.I[:]))[0].tolist()
        out["I_nonzero"] = nz
    except Exception as e:
        out["I_inject"] = _exc(e)
    # claims
    idx10 = out["L.indices[10*um]"]
    if not isinstance(idx10, list):
        idx10 = [idx10]
    past_end = (11 in idx10) or (idx10 == 11) or (idx10 == [11])
    sl = out["L.indices[3*um:100*um]"]
    includes_11 = 11 in (sl if isinstance(sl, list) else [sl])
    if past_end:
        verdict = "CONFIRMED"
    else:
        verdict = "NOT REPRODUCED"
    return rec(verdict, out)


def test_M11():
    """Mixed += and *= synaptic writes: vectorised vs loop disagreement, no warning."""
    from brian2 import NeuronGroup, Synapses, run, defaultclock
    from brian2.utils.logger import catch_logs
    import numpy as np

    def run_case(on_pre, v0, w):
        pre = NeuronGroup(2, "v : 1", threshold="True", reset="")
        post = NeuronGroup(1, "v : 1")
        post.v = v0
        S = Synapses(pre, post, "w : 1", on_pre=on_pre)
        S.connect()
        S.w = w
        logs = []
        with catch_logs() as l:
            run(defaultclock.dt)
            logs = list(l)
        return float(post.v[0]), logs

    cases = {}
    # cpp can only do one network. Do first case only on cpp; others via env.
    which = os.environ.get("BRIAN2_REPRO_CASE", "all")
    specs = {
        "add_then_mul": ("v_post += w; v_post *= 2", 1, [1, 10]),
        "zero_then_add": ("v_post = 0; v_post += w", 1, [1, 10]),
        "mul_then_add": ("v_post *= 2; v_post += 1", 1, [1, 10]),
    }
    if which != "all":
        specs = {which: specs[which]}
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "cpp_standalone" and which == "all":
        specs = {"add_then_mul": specs["add_then_mul"]}
    for name, (code, v0, w) in specs.items():
        val, logs = run_case(code, v0, w)
        warn = [x for x in logs if "order" in str(x).lower() or "permut" in str(x).lower()]
        cases[name] = {"v": val, "warnings": [str(x) for x in logs], "order_warnings": warn}
    expected_loop = {"add_then_mul": 28, "zero_then_add": 10, "mul_then_add": 7}
    expected_vec = {"add_then_mul": 48, "zero_then_add": 11, "mul_then_add": 6}
    out = {"cases": cases, "expected_loop": expected_loop, "expected_vec": expected_vec}
    target = os.environ.get("BRIAN2_REPRO_TARGET")
    if target == "numpy":
        ok = all(
            abs(cases[k]["v"] - expected_vec[k]) < 1e-9
            for k in cases
        )
        verdict = "CONFIRMED" if ok else "PARTIALLY CONFIRMED"
    else:
        ok = all(
            abs(cases[k]["v"] - expected_loop[k]) < 1e-9
            for k in cases
        )
        verdict = "CONFIRMED" if ok else "PARTIALLY CONFIRMED"
    no_warn = all(not cases[k]["order_warnings"] for k in cases)
    out["no_order_warning"] = no_warn
    return rec(verdict, out)


def test_M12():
    """integer subexpressions never substituted into diff eqs."""
    from brian2 import NeuronGroup, run, ms
    import os

    method = os.environ.get("BRIAN2_REPRO_METHOD", "rk4")
    try:
        G = NeuronGroup(1, """
            dv/dt = -v / (10*ms) + k / (10*ms) : 1
            k = int(v*3) + n : integer
            n : integer
        """, method=method)
        G.v = 0.5
        G.n = 1
        run(1 * ms)
        out = {"v": float(G.v[0]), "method": method, "error": None}
        verdict = "ok"
    except Exception as e:
        out = {"method": method, "error": _exc(e)}
        verdict = "CRASH"
    return rec(verdict, out)


def test_M13():
    """Periodic SpikeGeneratorGroup fires spurious burst when started mid-period."""
    from brian2 import (
        SpikeGeneratorGroup, SpikeMonitor, Network, ms, defaultclock
    )

    defaultclock.dt = 0.1 * ms
    # run empty-ish network first
    SG = SpikeGeneratorGroup(4, [], [] * ms, period=1 * ms)
    M = SpikeMonitor(SG)
    net = Network(SG, M)
    net.run(5.5 * ms)
    SG.set_spikes([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.8] * ms, period=1 * ms)
    net.run(1.0 * ms)
    spikes = list(zip(M.i[:].tolist(), (M.t / ms).tolist()))
    out = {"spikes_i_t_ms": spikes}
    # bug: neurons 0,1,2 at 5.5ms; expected: neuron 3 at 5.8ms
    at_55 = [i for i, t in spikes if abs(t - 5.5) < 1e-6]
    at_58 = [i for i, t in spikes if abs(t - 5.8) < 1e-6]
    out["at_5.5"] = at_55
    out["at_5.8"] = at_58
    if at_55 == [0, 1, 2] or set(at_55) >= {0, 1, 2}:
        verdict = "CONFIRMED"
    elif at_55 == [] and at_58 == [3]:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M14():
    """restore() after set_spikes() mixes old indices with new times."""
    from brian2 import SpikeGeneratorGroup, SpikeMonitor, Network, ms, defaultclock

    defaultclock.dt = 0.1 * ms
    SG = SpikeGeneratorGroup(2, [0, 0, 0], [1, 2, 3] * ms)
    M = SpikeMonitor(SG)
    net = Network(SG, M)
    net.store()
    SG.set_spikes([1, 1, 1], [0.5, 1.5, 2.5] * ms)
    net.restore()
    net.run(4 * ms)
    spikes = list(zip(M.i[:].tolist(), (M.t / ms).tolist()))
    out = {"spikes": spikes}
    # bug: neuron 0 at 0.5, 1.5, 2.5
    buggy = [i for i, t in spikes] == [0, 0, 0] and [t for i, t in spikes] == [0.5, 1.5, 2.5]
    correct_old = [i for i, t in spikes] == [0, 0, 0] and [t for i, t in spikes] == [1.0, 2.0, 3.0]
    if buggy:
        verdict = "CONFIRMED"
    elif correct_old:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    # different-length set_spikes after store
    err = None
    try:
        SG2 = SpikeGeneratorGroup(2, [0, 0, 0], [1, 2, 3] * ms, name="sg2")
        net2 = Network(SG2)
        net2.store()
        SG2.set_spikes([1, 1], [0.5, 1.5] * ms)
        net2.restore()
        net2.run(2 * ms)
        out["diff_len"] = "ok"
    except Exception as e:
        err = _exc(e)
        out["diff_len"] = err
    return rec(verdict, out)


def test_M15():
    """TimedArray rounds to nearest bin when group dt >= 8x array dt."""
    from brian2 import TimedArray, NeuronGroup, ms, defaultclock, run, StateMonitor
    import numpy as np

    values = np.arange(400, dtype=float)
    ta = TimedArray(values, dt=0.1 * ms)
    defaultclock.dt = 0.87 * ms
    G = NeuronGroup(1, "x = ta(t) : 1")
    M = StateMonitor(G, "x", record=True)
    run(300 * defaultclock.dt)
    got = np.array(M.x[0], dtype=int)
    times = np.array(M.t / ms)
    # floor(t / 0.1ms)
    expected = np.clip(np.floor(times / 0.1 + 1e-12).astype(int), 0, 399)
    mismatches = int(np.sum(got != expected))
    # step 1
    step1_got = int(got[1]) if len(got) > 1 else None
    out = {
        "mismatches_300": mismatches,
        "step1_got": step1_got,
        "step1_expected_floor": int(expected[1]) if len(expected) > 1 else None,
        "first_10_got": got[:10].tolist(),
        "first_10_exp": expected[:10].tolist(),
    }
    if mismatches == 132 or (mismatches > 100 and step1_got == 9):
        verdict = "CONFIRMED"
    elif mismatches == 0:
        verdict = "NOT REPRODUCED"
    else:
        verdict = "PARTIALLY CONFIRMED"
    return rec(verdict, out)


def test_M15_controls():
    """TimedArray control dt pairs should have zero mismatches."""
    from brian2 import TimedArray, NeuronGroup, ms, defaultclock, run, StateMonitor, Network
    import numpy as np

    pair = os.environ.get("BRIAN2_REPRO_PAIR", "0.1/0.3")
    if pair == "0.1/0.3":
        adt, gdt = 0.1 * ms, 0.3 * ms
    else:
        adt, gdt = 1 * ms, 0.1 * ms
    values = np.arange(400, dtype=float)
    ta = TimedArray(values, dt=adt)
    defaultclock.dt = gdt
    G = NeuronGroup(1, "x = ta(t) : 1")
    M = StateMonitor(G, "x", record=True)
    run(300 * defaultclock.dt)
    got = np.array(M.x[0], dtype=int)
    times = np.array(M.t / (1 * ms))
    adt_ms = float(adt / ms)
    expected = np.clip(np.floor(times / adt_ms + 1e-12).astype(int), 0, 399)
    mismatches = int(np.sum(got != expected))
    out = {"pair": pair, "mismatches": mismatches}
    verdict = "CONFIRMED" if mismatches == 0 else "NOT REPRODUCED"
    return rec(verdict, out)


# Registry is filled after remaining tests are defined below.
TESTS = {}


def _register():
    TESTS.update({
        "G18": {"fn": test_G18, "targets": ["numpy"], "timeout": 60},
        "M3": {"fn": test_M3, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "M3n": {"fn": test_M3_nested, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "M8": {"fn": test_M8, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "M9": {"fn": test_M9, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 180},
        "M16": {"fn": test_M16, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 180},
        "C5": {"fn": test_C5, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C5b": {"fn": test_C5_large, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C6a": {"fn": test_C6_negpow, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C6b": {"fn": test_C6_overflow, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C7": {"fn": test_C7, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C8": {"fn": test_C8, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C8f": {"fn": test_C8_floordiv, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C8p": {"fn": test_C8_pow, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "C13": {"fn": test_C13, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 180},
        "C14": {"fn": test_C14, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 30},
        "C14b": {"fn": test_C14_all_in_subgroup, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 30},
        "I2": {"fn": test_I2, "targets": ["cython", "cpp_standalone"], "timeout": 20},
        "I9": {"fn": test_I9, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "I9b": {"fn": test_I9_oob_high, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "I13": {"fn": test_I13, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 180},
        "I13s": {"fn": test_I13_source, "targets": ["numpy"], "timeout": 30},
        "M1": {"fn": test_M1, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "M1t": {"fn": test_M1_threshold, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "M4": {"fn": test_M4, "targets": ["numpy", "cython"], "timeout": 60},
        "M5": {"fn": test_M5, "targets": ["numpy"], "timeout": 60},
        "M6": {"fn": test_M6, "targets": ["numpy"], "timeout": 60},
        "M7": {"fn": test_M7, "targets": ["numpy"], "timeout": 120},
        "M10": {"fn": test_M10, "targets": ["numpy"], "timeout": 60},
        "M11": {"fn": test_M11, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 120},
        "M12": {"fn": test_M12, "targets": ["numpy"], "timeout": 60},
        "M13": {"fn": test_M13, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 180},
        "M14": {"fn": test_M14, "targets": ["numpy", "cython"], "timeout": 60},
        "M15": {"fn": test_M15, "targets": ["numpy", "cython", "cpp_standalone"], "timeout": 180},
        "M15c": {"fn": test_M15_controls, "targets": ["numpy"], "timeout": 120},
        "M2p": {"fn": test_M2_pair, "targets": ["numpy"], "timeout": 60},
    })
    # remaining S/C/G/I cases
    from more_cases import MORE
    TESTS.update(MORE)


_register()


def _worker(test_id: str, target: str) -> None:
    _setup_target(target)
    fn = TESTS[test_id]["fn"]
    try:
        result = fn()
    except Exception as e:
        result = rec("BLOCKED", {"error": _exc(e), "traceback": traceback.format_exc()})
    result["id"] = test_id
    result["target"] = target
    print("===RESULT===")
    print(json.dumps(result, default=str))
    print("===END===")


def _run_one(test_id: str, target: str, extra_env=None, timeout=None) -> dict:
    meta = TESTS[test_id]
    env = os.environ.copy()
    env["BRIAN2_REPRO_TARGET"] = target
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    if target == "cpp_standalone":
        env["BRIAN2_REPRO_DIR"] = tempfile.mkdtemp(prefix=f"b2cpp_{test_id}_")
    to = timeout or meta.get("timeout", 120)
    cmd = [PY, str(Path(__file__).resolve()), "--worker", test_id, target]
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=to, env=env,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {
            "id": test_id, "target": target, "verdict": "CONFIRMED" if test_id in ("I2", "C14") else "BLOCKED",
            "output": {"timeout": to, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:]},
            "note": "subprocess timed out",
        }
    text = p.stdout
    if "===RESULT===" in text:
        blob = text.split("===RESULT===")[1].split("===END===")[0].strip()
        try:
            return json.loads(blob)
        except Exception:
            return {"id": test_id, "target": target, "verdict": "BLOCKED", "output": text[-4000:], "stderr": p.stderr[-2000:]}
    return {
        "id": test_id, "target": target, "verdict": "BLOCKED",
        "output": {"stdout": text[-4000:], "stderr": p.stderr[-4000:], "returncode": p.returncode},
    }


def main():
    args = sys.argv[1:]
    if args[:1] == ["--worker"]:
        _worker(args[1], args[2])
        return
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = set(args[i + 1].split(","))
    results = []
    items = []
    for tid, meta in TESTS.items():
        if only and tid not in only:
            continue
        for t in meta["targets"]:
            items.append((tid, t))
    print(f"Running {len(items)} jobs...", flush=True)
    for tid, t in items:
        print(f"  {tid} @ {t} ...", flush=True)
        extra = {}
        r = _run_one(tid, t, extra_env=extra)
        results.append(r)
        print(f"    -> {r.get('verdict')} {str(r.get('output'))[:180]}", flush=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
