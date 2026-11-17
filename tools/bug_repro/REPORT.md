# Brian2 bug reproduction pass — round 2

Nothing was fixed. This is an independent reproduction of the 2026-11-13 audit
(`BRIAN2` findings M1–M16, S1–S13, C1–C15, G1–G18, I1–I13), with Cython and
C++ standalone actually compiled and run.

## Environment

| | Original audit | This pass |
|---|---|---|
| Commit | `3f1ddd55` (`2.11.0-46-g3f1ddd55`) | `d8196e19` (`2.11.0-50-gd8196e19`, latest `upstream/master` as of 2026-11-16) |
| OS | Windows 11 | Amazon Linux 2023, x86_64 |
| Python | 3.12.6 | 3.13.17 |
| numpy | 2.4.2 | 2.5.4 |
| Compiler | none (NumPy only; C++/Cython by reading) | gcc/g++ 11.5.0, Cython 3.3.1, scipy 1.18.1 |
| Targets | numpy | **numpy, cython, cpp_standalone** |

Four commits separate the two tips (`#1925` Cython cache-key work). No
behavioural change in the audited areas was expected from those commits.

Harness: `tools/bug_repro/run_repro.py` (each `(id, target)` is a fresh
subprocess).

## G18 — Priority 0 resolution

**CONFIRMED on this environment. The original contradiction is numpy-version
dependent, not a fluke of the first audit pass.**

```python
G = NeuronGroup(1, 'w : volt'); G.w = 1*volt
np.sin(G.w)
# array([0.84147098])   # plain ndarray, units dropped
```

| Call | Result here (numpy 2.5.4 / Python 3.13) |
|---|---|
| `np.sin(G.w)` | silent `ndarray` `[0.84147098]` (sin of 1 radian-as-if-dimensionless) |
| `np.cos` / `np.exp` / `np.sqrt` / `np.log` / `np.tanh` | same: unitless ndarray |
| `G.w.__array__ufunc__(np.sin, "__call__")` | **raises** `DimensionMismatchError: sin expects a dimensionless argument but got [1.] V` |

`VariableView` defines the misspelled `__array__ufunc__` (variables.py:1415)
and does **not** define `__array_ufunc__`. NumPy never calls the misspelled
method. Direct invocation of that dead method *does* go through `self[:]`
(a `Quantity`) and then raises — which is what the original independent
re-run likely hit on numpy 2.4.2, or they evaluated `np.sin(G.w[:])`.

On numpy 2.5.4, `np.sin(G.w)` converts via `VariableView.__array__` to a
raw ndarray and drops units. **The misspelling is the bug**; the first-pass
claim is true on current NumPy. The second-pass `DimensionMismatchError` was
a different call path, not a refutation.

---

## Per-item results

Format: `ID | verdict | targets | evidence | severity`

Severity: **silent-wrong** / **crash** / **hang** / **target-disagree** /
**api-footgun** / **cosmetic**.

### Priority 1 — C++/Cython claims that had never been compiled

| ID | verdict | targets | evidence | severity |
|---|---|---|---|---|
| **M3** | **CONFIRMED** all three | numpy, cython, cpp | `A=2*v+1`; `v=A+10; z=A` → `v=13, z=3` (expected 27). Nested `B=2*v; A=B+1`; `y=A; v=10; z=A` → `z=3` (expected 21). Same on all targets. | silent-wrong |
| **M8** | **CONFIRMED** (and C++/Cython copy as claimed) | numpy, cython, cpp | numpy: `v` becomes `[2,3,4]`. cython **and** cpp: `v` stays `[1,2,3]`, `x=[2,3,4]`. Cython also copies (audit only claimed C++). | silent-wrong / target-disagree |
| **M9** | **CONFIRMED** numpy; **C++/Cython correct** (now actually run) | numpy, cython, cpp | `n='1+j%2'` with `j='N_post-1-k for k in range(N_post) if i==0'`. numpy counts-by-j `{0:2,1:1,2:2,3:1,4:2,5:1}` (even targets doubled). cython/cpp `{0:1,1:2,2:1,3:2,4:1,5:2}` (correct). `j='k for k in range(3,6)'` raises on numpy (`_n[_j]`). | silent-wrong |
| **M16** | **CONFIRMED** cython (compiled); numpy/cpp correct | numpy, cython, cpp | `BinomialFunction(8, 0.6, approximate=False)`, N=20000. cython mean **5.7917**; numpy **4.8018**; cpp **4.8018**. True mean 4.8. This is no longer “generated-code only”. **Already filed: #1890.** | silent-wrong |
| **C5** | **CONFIRMED** cpp; numpy/cython stay double | numpy, cython, cpp | `i/(i+1)` at i=2: numpy/cython `0.6666666666666666`; cpp `0.6666666865348816` (= float32 `2/3`). k=2^24+1: cpp matches float32 `0.9999998807907104`; numpy/cython match float64. | silent-wrong |
| **C6** | **CONFIRMED** three-way split on `run_regularly`; assignment path differs | numpy, cython, cpp | `run_regularly('v = 2**(-k)')` with k=2: numpy `ValueError: Integers to negative integer powers are not allowed`; cython **`v=0.0`**; cpp **`v=0.25`**. String assignment `G.v='2**(-k)'` on cython *also raises* (numpy integer-power), so the `__Pyx_pow_long`→0 path is the compiled loop, not every assignment. `i**3` at i=1291: numpy **and cython** overflow int32 to `-2143282125`; cpp is correct `2151685171`. **Correction:** cython overflows like numpy, not like C++. | target-disagree |
| **C7** | **CONFIRMED** cpp; **NOT REPRODUCED on cython** | numpy, cython, cpp | `clip(i, 0.5, 2.5)` → numpy `[0.5,1,2,2.5,2.5]`; **cython same as numpy**; cpp `[0,1,2,2,2]`. Audit expected cython to truncate; compiled cython does not. | target-disagree |
| **C8** | **PARTIALLY CONFIRMED** | numpy, cython, cpp | `k%=3` with k=-2: numpy **1**, cython **1**, cpp **-2** (C remainder). `//=` and `**=` are **not** emitted to C++: they fail earlier with `AssertionError: Unknown operator "//="` / `"**="` on all targets, including cpp. Audit’s “//= starts a comment” / “**= is a C++ syntax error” is **wrong** on current master — the Python translator rejects them first. | target-disagree / api-footgun |
| **C13** | **CONFIRMED** numpy; cython/cpp also error | numpy, cython, cpp | `connect(p=-0.1)`: N=6 → 0 synapses (numpy); N=2000 → `ValueError` via `_sample_without_replacement` (negative dimensions). Cython errors even for N=6. | api-footgun |
| **C14** | **NOT REPRODUCED as a hang**; source defect still present | numpy, cython, cpp | SpikeMonitor on `G[:5]` (`_start_idx==0`) with only neurons 5–9 spiking: all three targets terminate, 0 subgroup spikes, 5 parent spikes. Everyone-spikes case records `[0,1,2,3,4]`. The `size_t _j>=_start_idx` loop is still in `spikemonitor.cpp:25`. A hang was not observed in a normal run (audit already noted it “usually ends with zero events”). | by-reading / latent UB |
| **I2** | **CONFIRMED hang** on both compiled targets | cython, cpp | `BinomialFunction(80, 0.5, approximate=False)`: **timed out at 20 s** on cython and cpp_standalone (stdout empty). Numpy is not in this test; it is known-fine. **Already filed: #1890 (item 2).** | hang |
| **I9** | **CONFIRMED** | numpy, cython, cpp | 2D `TimedArray` column `-1` inside equations: numpy wraps to last column (`1.0`); cython/cpp return **NaN**. Column `2`: numpy `IndexError`; cython/cpp **NaN**. Python-level `ta(0*ms, -1)` wraps on all (the pyfunc). | target-disagree |
| **I13** | **CONFIRMED by generated source**; runtime too noisy to see float32 loc | numpy, cython, cpp | Generated: `cdef float bino(...)` and `float bino(...)`. 5000 samples of N(1234568, 1040) cannot distinguish float32 vs float64 loc (difference 0.013). **Related: #1898** (float vs int32 assignment). | silent-wrong (decl) |

### Priority 2 — major bugs

| ID | verdict | targets | evidence | severity |
|---|---|---|---|---|
| **M1** | **CONFIRMED all three** (folding is pre-codegen) | numpy, cython, cpp | `G.v = 'i * 10**-3'` → `[0,0,0,0,0]`. Controls: `2.0**-1` → 0.5; `i * 1/10**3` → `[0, 0.001, …]`. Threshold/reset with `G.v=1`: **1 spike**, not 200 — reset folds to 0 so `v>0` is then false. The “200 spikes in 20 ms” recipe is **not** reproduced with v starting at 0 or 1; the assignment fold is. | silent-wrong |
| **M2** | **NOT REPRODUCED** on the docs concatenation pattern | numpy | `common = Equations('dv/dt = -(v+I)/tau : volt')`; `eqs1 = common + Equations('I = 10*mV : volt')`; with and without a second `eqs2 = common + Equations('I : volt')`, `method='exact'` both give `-32.072766 mV` matching the analytic `-10-60*e^-1`. Same for `exponential_euler`. `rk4` did **not** KeyError. `euler` is just integrator error. The shared-`SingleEquation` mutation may still exist for a narrower pattern; it does not fire on the documented example here. | — |
| **M4** | **CONFIRMED** | numpy, cython | `SG=G[2:7]; SG.v[1:3]='i+100'` writes neurons **5,6** as 103,104 (should be 3,4). `SG.v[3:5]='1'` raises (IndexError wrapped in BrianObjectException). | silent-wrong |
| **M5** | **CONFIRMED** | numpy | `SG=G[2:7]; SG3=SG[1:3]`; `SG.v[SG3]` → `[5,6]` not `[3,4]`; write `-1` overwrites neurons 5,6. `G[5:10].v[G[5:7]]` → IndexError. `G.v[SG]` on the parent is correct. | silent-wrong |
| **M6** | **CONFIRMED** | numpy | `dv/dt=(v-v**2)/tau`, v0=0.1, T=2τ: exponential_euler **0.7389056** (= `0.1*e^2` = linear `v/tau`). scipy `solve_ivp`: **0.450853**. Does not raise. | silent-wrong |
| **M7** | **CONFIRMED** | numpy | Documented `refractory='(1+2*rand())*ms'`, 2000 neurons × 100 ms: mean ISI **1.430 ms** (doc 2.0), fraction >2 ms **1.76%** (uniform 50%). Matches the audit (1.43 ms / 1.5%). **No upstream issue found** for this (search: refractory+rand; #41 is the original refractoriness design). | silent-wrong |
| **M10** | **CONFIRMED** (indices); I injection API differs | numpy | Soma + L (10 µm, n=10) + L.L (n=5). `m.L.indices[10*um]` → **11**; `[3*um:100*um]` → `[4..11]`; `[-20]` → **-9**. `neuron.I[...]` is not an attribute on this model (`AttributeError`); the out-of-section index 11 is the real bug. | silent-wrong |
| **M11** | **CONFIRMED** all three; no warning | numpy, cython, cpp | v0=1, w=[1,10], two spikes onto one target. `v+=w; v*=2`: numpy **48**, cython **28**, cpp **28**. `v=0; v+=w`: numpy **11**, cython **10**. `v*=2; v+=1`: numpy **6**, cython **7**. `catch_logs` empty in every case. Related (not the same): open **#1110** (numpy vectorises synaptic events, cython does not). | silent-wrong / target-disagree |
| **M12** | **PARTIALLY CONFIRMED** (rk2/rk4 crash; numeric values not the audit’s 0.36366) | numpy | `k = int(v*3)+n : integer` in an ODE: `rk2`/`rk4` **CRASH** (`BrianObjectException` wrapping the expected KeyError path). `exact` and `exponential_euler` **run** and agree with each other (v≈0.643 on this ODE), i.e. they accept the system instead of substituting the integer subexpression. Did not reproduce the 0.36366 vs 0.36836 pair (different right-hand side than the audit’s unpublished script). | crash / silent-wrong |
| **M13** | **CONFIRMED** numpy+cython; cpp blocked on multi-run | numpy, cython | After `run(5.5*ms)`, `set_spikes([0,1,2,3],[0.1,0.2,0.3,0.8]*ms, period=1*ms)`: spikes at 5.5 ms are neurons **0,1,2**; neuron 3 at 5.8. **Already filed: #1936.** cpp_standalone refused a second `run()` (`build_on_run` default). | silent-wrong |
| **M14** | **CONFIRMED** | numpy, cython | store → `set_spikes([1,1,1],[0.5,1.5,2.5]*ms)` → restore → run records neuron **0** at 0.5, 1.5, 2.5 ms. Different-length `set_spikes` then restore errors. **Already filed: #1932.** | silent-wrong |
| **M15** | **CONFIRMED** rounding vs floor; count is not 132 | numpy, cython, cpp | array dt 0.1 ms, clock 0.87 ms. Step 1: got **9**, floor-expected **8**. Mismatches in 300 steps: 18 (numpy) / 22 (cython, cpp), not 132 — the rounding bug is real, the 132 figure depends on how “expected” is computed. Control 0.1/0.3 ms: **0 mismatches**. | silent-wrong |

### Priority 3 — spatial / morphology

| ID | verdict | targets | evidence | severity |
|---|---|---|---|---|
| **S1** | **CONFIRMED** | numpy | no coords: `TypeError: The length argument has to be a single value.` with coords: `DimensionMismatchError` (bare floats vs metres). | crash |
| **S2** | **CONFIRMED** | numpy | non-Soma root `generate_coordinates()` → `ZeroDivisionError: float division by zero`. | crash |
| **S3** | **CONFIRMED** | numpy | `n.R1[:]` → `KeyError: 'R'`; `n.L[:]` → `AttributeError: The subtree L does not exist`; with an L child, `n.L` is a `SpatialSubgroup`. | api-footgun |
| **S4** | **CONFIRMED** | numpy | `from_swc_file(..., spherical_soma=False)` on 3-point soma, 1-point soma, and 1-point non-soma, plus `from_points`: all `ValueError: The number of compartments n has to be at least 1.` | crash |
| **S5** | **CONFIRMED** | numpy | `SpatialNeuron(m.axon, ...)` with N=16: `n.main` starts at **1**, `n.L` is 14–17 (outside the group). | silent-wrong |
| **S6** | **CONFIRMED** | numpy | `neuron.L[:5*um]` → `TypeError: only integer scalar arrays...`; `[3*um:]` TypeError on None dimensions; `[5*um]` TypeError. Morphology forms work. | api-footgun |
| **S7** | **CONFIRMED** | numpy | `del s.axon` then `s['1']` still returns the Cylinder; `del s['1']` → `ValueError: list.remove(x): x not in list`. | api-footgun |
| **S8** | **CONFIRMED** | numpy | `_perturb` along x=y: mean angle **1.09e-7°** (requested 20°). Deterministic branch-angle half of the test hit a `DimensionMismatchError` in `generate_coordinates` (related to S1). | silent-wrong |
| **S9** | **CONFIRMED** | numpy | single `diameter`+`length`: TypeError (wants length-4 array). `n=2.5` stored as **2**. Mutating the input `x` array: stored x[0] became 0.5 (compartment centre) — input was not a defensive copy of the endpoints in the way the audit described; still a constructor mismatch. | api-footgun |
| **S10** | **CONFIRMED** | numpy | `pickle.dumps(morphology)` → `AttributeError: Can't get local object 'Children.__init__.<locals>.<lambda>'`. `copy`/`deepcopy` ok. | crash |
| **S11** | **CONFIRMED** | numpy | `threshold_location=100` on 23 compartments, and `1.5`, both **create** a SpatialNeuron with no error. | api-footgun |
| **S12** | **PARTIALLY CONFIRMED** | numpy | attaching a child named `"11"` then `morpho['11']` returned that Cylinder (n=1) in this setup — did not reproduce “parsed as child 1 of child 1”. The `LL` path parsing *is* real (it broke the first M10 harness). | api-footgun |
| **S13** | **CONFIRMED** | numpy | unknown SWC type 99: `topology()` prints `.None`; child name is `None`. | cosmetic |

### Priority 4 — remaining codegen

| ID | verdict | targets | evidence | severity |
|---|---|---|---|---|
| **C1** | **CONFIRMED** | numpy, cpp | `floor(x*1e10)` with x=1: numpy **-2147483648** (int32 min); cpp **10000000000.0** (double). | silent-wrong / target-disagree |
| **C2** | **CONFIRMED** | numpy | `G.indices['v>1']` dtype **float64**; `G.v[G.indices['i<3']]` TypeError. Shared-variable string index also wrong-shaped / TypeError. | api-footgun |
| **C3** | **PARTIALLY CONFIRMED** | numpy | `int(b)*exp(v)+rand()` → `rand_func() missing 1 required positional argument: 'vectorisation_idx'`. Still fails with optimisation **off**, so it is not only the `where()` shortcut. | api-footgun |
| **C4** | **NOT REPRODUCED** | numpy | `S.connect('v_post > 1', skip_if_invalid=True)` created 15 synapses. The shape-mismatch may need a generator-style `j='...'` condition; the simple string condition works. | — |
| **C9** | **CONFIRMED** | numpy | After `prefs.codegen.loop_invariant_optimisations = False`, a **new** group with `i*10**-3` still folds to zeros (M1 cache). | silent-wrong |
| **C10** | **CONFIRMED** | numpy | `x/0.0`, `x/0`, `1e400*x` all become `v = inf * x` then `NameError: name 'inf' is not defined`. `x*inf*0` folds to **0**; `(x-1)/(x-1)` at x=1 folds to **1**. | crash / silent-wrong |
| **C11** | **CONFIRMED** | numpy | `G.v='2**tau'` with `tau=10*ms` runs (`1.00696`). `Equations('w = v**tau : volt**0.01')` accepted (`V^0.01`). | api-footgun |
| **C12** | **NOT REPRODUCED** on this path | numpy | `Equations('dv/dt = (0.30000000000000004 - v)/second : 1')` **keeps** `0.30000000000000004`. `sympy.sympify` of that literal also keeps it on sympy 1.14. The 15-digit trim may be a narrower renderer path. | — |
| **C15** | **CONFIRMED** by reading | — | `codeobject.py:186` still emits the literal `'code_lines[0]'` (missing `f`). | cosmetic |

### Priority 5 — groups / equations / state updaters

| ID | verdict | targets | evidence | severity |
|---|---|---|---|---|
| **G1** | **CONFIRMED** | numpy | `a='b', b='a'` → `(a - a)`; `a='b', b='c'` → `(c + c)`; `tau='tau_e + tau_i'` → `-v/tau_e + tau_i` (no parens). | silent-wrong |
| **G2** | **CONFIRMED** | numpy | `threshold=Expression('v > 1')` → `TypeError: argument of type 'Expression' is not iterable` (`"rand(" in threshold`). | crash |
| **G3** | **CONFIRMED** | numpy | `hash(Equations('v : 1'))` and `hash(SingleEquation)` → `TypeError: unhashable type: 'list'`. | crash |
| **G4** | **CONFIRMED** | numpy | `G.get_states(['sub'])` and `set_states({'w':'sub+5'})` → `KeyError: identifier "ext"`. `G.sub[:]` is a different error here (0-d vs 1-d) but the `level` argument still does not pull in locals. | api-footgun |
| **G5** | **BLOCKED** | numpy | TimedArray + `method='exact'` raised on the dt=0.5 ms group in this harness (BrianObjectException). Did not get a clean refused-then-cached-accept sequence. | — |
| **G6** | **CONFIRMED** | numpy | `G.v='1.e3 * i'` → `KeyError: identifier "e3"`. | crash |
| **G7** | **CONFIRMED** | numpy | user `events` dict gains `'spike'`; `dtype` gains `'lastspike'`; second NeuronGroup with the same dict → `ValueError: defines both a threshold and a 'spike' event`. | api-footgun |
| **G8** | **CONFIRMED** | numpy | `legacy.refractory_timing=True` + string refractory: parse fails on `(t - lastspike) > (1+rand())*ms` because the generated line lacks `not_refractory =`. | crash |
| **G9** | **CONFIRMED** | numpy | `method=['exact','euler'], method_options={'bogus':1}` accepted. | api-footgun |
| **G10** | **CONFIRMED** | numpy | `G.s.shape` on a shared variable → bare `AssertionError`. | crash |
| **G11** | **CONFIRMED** | numpy | linked to size-1 group: `Gb.q[:]` is numpy **scalar** `3.0`; `Gb.q[1]` → `AttributeError: 'str' object has no attribute 'get_value'`. | api-footgun |
| **G12** | **PARTIALLY CONFIRMED** | numpy | linking through a subexpression raised `KeyError: identifier "y"` (not `_x_indices`). Same class of failure, different key. | crash |
| **G13** | **CONFIRMED** | numpy | LaTeX repr: `(unit: $\mathrm{rad}$)` for `x : 1`. | cosmetic |
| **G14** | **CONFIRMED** | numpy | `str(Equations('b : boolean\nk : integer\nv : volt'))` → `b : 1\nk : 1\nv : V`. Round-trip `Equations(str(Equations('v : volt')))` → `EquationError`. | api-footgun |
| **G15** | **CONFIRMED** | numpy | `Equations('_x : 1')` rejected; `Equations('v : 1') + '_x : 1'` accepted. Same for `ms`. | api-footgun |
| **G16** | **CONFIRMED** | numpy | `G.v[np.uint32(3)]` and `G.v[[]]` → TypeError (signed-integer-only check). | api-footgun |
| **G17** | **CONFIRMED** by reading | — | `__imod__` error text mentions `//`; C15-style missing f-prefix still in `codeobject.py`. | cosmetic |
| **G18** | **CONFIRMED** | numpy | see Priority 0. | silent-wrong |

### Priority 6 — input / IO / prefs

| ID | verdict | targets | evidence | severity |
|---|---|---|---|---|
| **I1** | **CONFIRMED** numpy; cython count differs | numpy, cython | N=3, spikes `[0,1,0]` at 0, 0.01, 0.02 ms, dt=0.1 ms: monitor `i=[0,1,0]`, numpy `count=[1,1,0]` (duplicate hidden), `v=[2,1,0]`. Cython `count=[2,1,0]` (count is right, check still didn’t raise). N=2: numpy SpikeMonitor errors (overflow); cython recorded `[0,1]` without error. | silent-wrong |
| **I3** | **CONFIRMED** by source | numpy | `poissoninput.py:84` still does `weight = repr(weight)`. Default Quantity repr is used in the generated statement. Harness constructor call was messy; the code path is unchanged. | silent-wrong |
| **I4** | **NOT REPRODUCED** | numpy | `PoissonInput(..., weight=np.float64(0.5))` on a dimensionless `v` **succeeded** on numpy 2.5.4. `repr(np.float64(0.5))` is still `'np.float64(0.5)'`; something in construction is no longer embedding that raw repr, or Quantity wrapping changes it. | — |
| **I5** | **NOT REPRODUCED** | numpy | `period=6.9*ms, dt=0.3*ms` **accepted** (`6.9/0.3 == 23.0` exactly here). Scan k=2..199 at dt=0.3 ms: **0** “not an integer multiple” rejects. Original 55 false rejects were on numpy 2.4.2 / Windows; this is a likely fp-environment difference. | — |
| **I6** | **CONFIRMED** | numpy | spike at 300000 s, dt=0.1 ms: recorded at **t=0**. | silent-wrong |
| **I7** | **CONFIRMED** | numpy | after `with catch_logs()`, a later `warn` is still appended (n_after_warn=2); handler count 3→4 on a second `with`. | leak |
| **I8** | **PARTIALLY CONFIRMED** | numpy | pandas export fails, but with `NotImplementedError: Units not supported when exporting to pandas data frame` rather than the audit’s TypeError / length-mismatch. Shared / SpikeMonitor / StateMonitor all fail. | api-footgun |
| **I10** | **CONFIRMED** | numpy | mutating a float64 input array changes `ta(0*ms)` (99); integer input is copied (stays 1). | silent-wrong |
| **I11** | **PARTIALLY CONFIRMED** | numpy | `prefs.codegen.target='nump'` is accepted. Setting the target to a nested class raised `PreferenceError` (not the `as_file` unreadable case). `default_integer_dtype=str` was accepted in the original recipe. | api-footgun |
| **I12** | **CONFIRMED** | numpy | `SpikeGeneratorGroup(3, [0.5, 1], ...)` stores indices **`[0, 1]`**. `PoissonInput` with `rate*dt>1` ran without a Brian-level check. | silent-wrong |

---

## 1. CONFIRMED, ranked by severity (new vs already-known)

### Silent wrong results — genuinely new (no matching open issue found)

1. **M1** — `10**-3` folded to 0, all targets (fold is in the optimiser, before codegen). Tiny reproducer.
2. **M3** — subexpression not recomputed after a same-line or nested write. **All three targets.**
3. **M4 / M5** — subgroup offset applied twice (string assignment and group-as-index).
4. **M6** — `exponential_euler` silently drops nonlinear terms.
5. **M7** — string refractory re-sampled every timestep (mean 1.43 ms vs documented 2.0).
6. **M8** — NumPy aliases `tmp=v` and in-place `+=` corrupts `v`; Cython/C++ copy.
7. **M9** — NumPy `connect(..., n=...)` indexes `_n` by `_j`; Cython/C++ correct (now proven by run).
8. **M10** — morphology distance index walks into the next section.
9. **M11** — mixed `+=` / `*=` synaptic writes: numpy 48 vs loop 28, **no warning**. Related to #1110 but not filed as this mixed-op case.
10. **C5** — C++ `1.0f*` int/int division.
11. **C1** — NumPy `floor`/`ceil` int32 overflow; C++ stays double.
12. **G1** — sequential keyword substitution (`(a-a)`, missing parens).
13. **C9** — `make_statements` cache ignores the optimisation pref (keeps M1 alive).
14. **I6** — spike times beyond 2^31 bins fire at t=0.
15. **G18** — `np.sin(G.w)` drops units (numpy 2.5).

### Silent wrong / hang — already upstream

| ID | upstream |
|---|---|
| **M16** Cython BinomialFunction p>0.5 | **#1890** (open, with patch described) |
| **I2** inversion hang for large n·p | **#1890** (same ticket, item 2) |
| **M13** periodic SpikeGeneratorGroup burst | **#1936** (open) |
| **M14** restore() mixes indices/times | **#1932** (open) |
| **I13** `float` approx declaration | adjacent to **#1898** |

### Crashes / leaks / API — new, good first issues

S1, S2, S4, S10, G2, G3, G6, G8, G10, I7, C10 (`inf` NameError), C2, G7, G14/G15, S3, S6, S11.

## 2. Where Cython/C++ now disagree with the original “by reading”

These are the valuable corrections from actually compiling:

| ID | Original claim | This pass |
|---|---|---|
| **C7** | clip(int,float,float) truncates on C++ **and Cython** | **Cython matches NumPy** (`[0.5,1,2,2.5,2.5]`). Only C++ truncates. |
| **C6** `i**3` | overflow is NumPy-only; C++ correct | **Cython also overflows int32** like NumPy. C++ is correct. |
| **C6** `2**(-k)` via assignment | Cython `__Pyx_pow_long` → 0 | String **assignment** on Cython raises like NumPy. The `0` result is only on **`run_regularly` / compiled loops**. C++ `pow` → 0.25 as claimed. |
| **C8** `//=` `**=` | emitted to C++, comment / syntax error | **Rejected in Python** (`AssertionError: Unknown operator`) before any C++ is emitted. `%=` still disagrees (numpy 1 vs cpp -2). |
| **C14** | size_t underflow hang when `_start_idx==0` | **Does not hang** on the constructed cases; returns the correct 0 events. Defect still in the source. |
| **M8** | only C++ copies | **Cython also copies.** NumPy is the odd one out. |
| **M16** | Cython wrong (emulation) | **Confirmed with a real Cython run**, mean 5.79 vs 4.80. C++ correct. |
| **I2** | hang (emulation) | **Confirmed: 20 s timeout** on both cython and cpp_standalone. |
| **I9** | C++/Cython return NaN (by reading) | **Confirmed by run.** |
| **I5** | 6.9 ms / 0.3 ms false-reject | **Accepted** on numpy 2.5.4 (23.0 exactly). Environment-specific. |
| **I4** | np.float64 weight fails under numpy 2 | **Accepted** here. |
| **M2** | shared Equations fragment corrupts first group | **Not reproduced** on the documented concatenation example. |
| **C12** | 0.30000000000000004 → 0.3 via sympy | **Not reproduced** on Equations / sympy 1.14. |
| **C4** | skip_if_invalid + post condition shape error | **Not reproduced** with a simple string condition. |

## 3. BLOCKED / incomplete

| ID | what’s missing |
|---|---|
| **M13** cpp | standalone refused a second `run()` after `set_spikes`. Cython already matches numpy. Would need `build_on_run=False` + `device.build()`. |
| **M11** cpp cases 2–3 | only `v+=w; v*=2` was run on cpp (one network per process). Cython covered all three. |
| **G5** | TimedArray + exact raised before the cache interaction could be shown. |
| **M12** numeric 0.36366 vs 0.36836 | rk2/rk4 crash confirmed; the exact scipy pair needs the audit’s unpublished ODE. |
| **M1** “200 spikes in 20 ms” | assignment fold confirmed; that specific threshold/reset recipe gives 1 spike when v starts at 0 or 1. |
| **S12** | `"11"` as a numeric name did not parse as 1-of-1 in this attachment method. |
| **I3** default-precision numeric error | `repr(weight)` path confirmed by reading; did not re-measure 1e-8 vs precision=2 in a live `PoissonInput` run. |

No compiler or library was missing. cpp_standalone and Cython both built successfully.

## 4. G18 (explicit)

On **numpy 2.5.4 / Python 3.13.17**, `np.sin(G.w)` on a volt `VariableView` **does not raise**. It returns `array([0.84147098])` with no units. The same for `cos`, `exp`, `sqrt`, `log`, `tanh`.

The misspelled `__array__ufunc__` is dead code. Calling it by hand raises `DimensionMismatchError` because it forwards to `self[:]` (a `Quantity`). That is the likely source of the original audit’s second-pass raise on numpy 2.4.2.

**Verdict: CONFIRMED (silent unit drop) on current NumPy; the misspelling is why.**

## Suggested filings (only items that are both CONFIRMED and not already upstream)

Highest leverage, small reproducer, local fix:

1. **M1** — `int ** int` typed as integer with a negative exponent.
2. **M6** — `exponential_euler` should raise if `collect` returns keys other than `var` and `1`.
3. **M4 / M5** — subgroup index offset applied twice.
4. **M9** — NumPy `.repeat(_n[_j])` → `.repeat(_n)`.
5. **M10** — upper bound on morphology distance indexing.
6. **M3** — subexpression invalidation on the same statement / nested.
7. **M8** — NumPy should `.copy()` read-only arrays that a later temp aliases.
8. **M7** — evaluate string refractory once per spike (or document the resampling).
9. **G18** — rename `__array__ufunc__` → `__array_ufunc__`.
10. **C5** — `1.0*` not `1.0f*` in C++ division.
11. **S2 / G2 / G3 / I7** — one-line crashes/leaks.

Do **not** re-file M13, M14, M16, I2 (#1936, #1932, #1890).
