# THA marginalized-SNR statistic: what it is and how to run it

This note covers the `marg_lnl` addition to `pycbc_inspiral_tha`: what it
actually computes, and how to build the one auxiliary input file it
needs before a production-scale search run. It assumes you already know
how to run a THA precessing-harmonic search; it does not re-explain the
existing pipeline (5-harmonic filtering, clustering, `snr_comp_1..5`,
etc.).

## 1. Background

`pycbc_inspiral_tha` filters each template against 5 orthogonal
precessing "harmonics" (per arXiv:1908.05707) and records the complex
per-harmonic SNR (`snr_comp_1..5`) at each trigger. The statistic used
to threshold/rank triggers today is the quadrature sum of these five
numbers. That statistic has a real weakness: it treats the 5 harmonics
as free, independent numbers and just adds up whatever magnitude shows
up. It has no way to check whether the *combination* of harmonic
amplitudes at a given sample is something a real signal could actually
produce, or whether it's just 5 unrelated numbers that happen to add up
to a large sum by chance.

`marg_lnl` is a new per-trigger column that fixes this: it is the log
of the likelihood, marginalized over the physical nuisance parameters
that determine how a real signal's power *should* be distributed across
the 5 harmonics, analytically marginalized over distance and phase as
well. Because it's evaluated against the actual physical manifold of
possible harmonic combinations, it naturally down-weights combinations
that don't look like any real precessing signal — something the
quadrature sum cannot do at all.

## 2. What `marg_lnl` actually is

For a given template's 5 orthonormal filters, any real signal observed
at that template's intrinsic parameters can be written as

```
h(f) = c1(θ)·h1(f) + c2(θ)·h2(f) + ... + c5(θ)·h5(f)
```

where `θ = (theta_jn, alpha0)` are two angles describing the
line-of-sight orientation relative to the system (`theta_jn`: analogous
to inclination; `alpha0`: precession phase at the reference frequency).
For any given `θ`, the observed data's overlap with that combination,
further combined with polarization angle `psi` (a third nuisance
parameter, entering only via a closed-form real rotation — no extra
waveform generation needed) and marginalized analytically over
luminosity distance and orbital phase, gives a marginalized
log-likelihood `lnL(θ, psi)`.

**Orbital phase (`phi0`)**: notice `phi0` is not a grid parameter, and
does not appear anywhere in the grid file. It doesn't need to be: `phi0`
enters every harmonic as an *exact* global phase,
`h(θ, phi0, psi) = exp(2j·phi0)·h(θ, 0, psi)` (checked directly during
development, not assumed — see §6). A phase shift of the template only
rotates the complex overlap `(d|h)` by `exp(2j·phi0)`, so the integral
`∫dphi0 exp(lnL(θ, phi0, psi))` depends only on `|(d|h)|` and can be done
in closed form via a modified Bessel function `I0` — this is exactly
what `LookupTableMarginalizedPhase22` (§3) already does (it's the same
analytic-phase-marginalization trick standard elsewhere in
PyCBC/cogwheel for quadrupole(`|m|=2`)-dominated signals, applied here
per-harmonic-combination instead of per-mode). Because of this,
`compute_raw_and_ab` generates each `(theta_jn, alpha0)` grid point's
`h+`/`hx` at a single, arbitrary, fixed `phi0` (hardcoded to `0`) with
no loss of generality — any other choice of `phi0` would just rotate
every grid point's waveform by the same overall phase, which the
`I0`-based marginalization integrates out exactly regardless, so there
is nothing to gain from making `phi0` a grid dimension.

`marg_lnl` is

```
marg_lnl = log ∫∫ p(θ, psi) exp(lnL(θ, psi)) dθ dpsi
```

i.e. the log-evidence, marginalized over `(theta_jn, alpha0, psi)` with
an isotropic-orientation prior, and analytically over distance and
phase (following the same closed-form distance-marginalization used in
`cogwheel`, adapted for this framework). In practice this integral is
done via a fixed quadrature grid in `(theta_jn, alpha0)` (locked in at
24×24, see §4) folded with a `psi` quadrature (16 points), combined via
`logsumexp`.

**What it is not**: it does not marginalize over sky location or time
of arrival (this is a single-detector statistic; those would need
independent handling), and it does not currently do a local
time-window search around the SNR-based trigger. Testing so far (a
handful of injections; see git history / conversation log for details)
found that for a template family with `num_comps ≤ 3` and the harmonic
generation bug described in §5 fixed, the sample with peak `marg_lnl`
essentially always coincides with the sample the existing
quadrature-sum clustering already selects — i.e. computing `marg_lnl`
only at existing trigger times looks safe so far — but this has only
been checked for a small number of injections and should not yet be
treated as settled for the general case, especially at low SNR or for
strongly-precessing (`num_comps` 4-5) templates.

## 3. Where the code lives

* `pycbc.filter.tha_marginalize` — runtime module (numpy/scipy only, no
  lalsimulation). Contains the distance/phase-marginalized likelihood
  lookup table, and the cheap per-segment machinery that turns
  PSD-independent per-template data into an actual `marg_lnl` value for
  a given PSD and a given `snr_comp_1..5` sample
  (`marginalize_for_psd`), plus the loader for the auxiliary grid file
  (`load_raw_grid_file`).
* `pycbc.waveform.tha_marg_grid` — offline module (uses
  `pycbc.waveform.bank`'s template classes, so it does need
  lalsimulation). Generates the 5 raw harmonics and the
  `(theta_jn, alpha0)` grid's expansion coefficients for one template
  (`compute_raw_and_ab`).
* `pycbc_make_tha_marginalization_grid` — the executable that builds
  the auxiliary input file over a bank (or a slice of one, for
  parallelizing across jobs — see §4). This is the "done once" step.
* `pycbc_combine_tha_marginalization_grid` — optional convenience
  executable that merges several split-job grid files into one (see
  §4); never required, since `pycbc_inspiral_tha` can load multiple
  grid files directly.
* `pycbc_inspiral_tha --marginalized-grid-file <path>` — the search-time
  flag that turns this on (accepts multiple paths for split-job grid
  files, see §4). Adds a `marg_lnl` column (float32, NaN for templates
  not present in the grid file) to the output trigger file. Immediately
  after loading, it checks each file's generation-provenance attrs
  (`pycbc.filter.tha_marginalize.load_raw_grid_file_attrs`) against this
  run's `--sample-rate`, resulting `delta_f` (`1 / --segment-length`),
  and `--low-frequency-cutoff`, and raises a `ValueError` naming the
  mismatch if they disagree (see §4).

## 4. Building the input file (done once)

```
pycbc_make_tha_marginalization_grid \
    --bank-file BANK.hdf \
    --output GRID.hdf \
    --sample-rate 2048 \
    --delta-f 0.00390625 \
    --low-frequency-cutoff 20 \
    --n-theta 24 --n-alpha 24 \
    --reference-psd-model aLIGOZeroDetHighPower
```

**`--sample-rate`/`--delta-f`/`--low-frequency-cutoff` must exactly
match the search's actual configuration** (i.e. what
`pycbc_inspiral_tha` will use for that run, via its own
`--sample-rate`/`--segment-length`/`--low-frequency-cutoff`). This is a
hard requirement, not a tuning knob: the grid file caches the 5 raw
harmonics at a specific frequency binning, and at search time they get
directly correlated against the segment's real PSD, so the array
lengths must line up. The grid file records these (plus `--interp` and
`--n-theta`/`--n-alpha`, and the approximant) as HDF5 attrs for exactly
this reason. `pycbc_inspiral_tha` checks them against its own options
immediately after loading each `--marginalized-grid-file` and raises a
clear `ValueError` naming the mismatching option(s) if they don't agree
(rather than silently producing garbage, or only failing deep inside
the per-segment filtering loop) — so a mismatch will not go unnoticed,
but building the file with the wrong `--delta-f` means rebuilding it.

**`--interp`**: by default, the grid is built with `interp=False`
(direct waveform generation at the target `--delta-f`, rather than
`bank.py`'s "generate coarse, then upsample" shortcut). Pass `--interp`
to use the shortcut instead — it is substantially cheaper, and was
validated (see §5) to be a fully usable option: running the same
injection through `pycbc_inspiral_tha` with an `--interp`-built grid
vs. a default (`interp=False`) grid gave `marg_lnl` values agreeing to
~1e-6 relative precision. `interp=False` remains the default because it
gives better internal numerical fidelity (~1e-14 vs ~1e-4, see §5) for
no meaningful cost difference at small scale; for very large banks
where build time matters more, `--interp` is a legitimate way to speed
this step up. Note this is independent of, and does not need to match,
anything about the real search's own filters: `bank.py`'s actual
per-segment filtering (`get_whitened_normalized_comps`) always
generates with `interp=True` internally, with no CLI override in
`pycbc_inspiral_tha` — the grid's `--interp` choice is purely an
internal precision/cost tradeoff for this one auxiliary file.

**`--reference-psd-model` does *not* need to match the real detector
noise**, and the grid file does *not* need rebuilding if the real PSD
changes (between segments, epochs, or detectors). This is the whole
point of the design: the file stores the 5 raw (unwhitened) harmonics
and the PSD-*independent* expansion coefficients of the orientation
grid in terms of those harmonics. `pycbc_inspiral_tha` re-derives the
actual PSD-dependent filters itself, per segment, from this cached
data, via cheap 5×5 linear algebra plus small matrix products — no
lalsimulation calls at search time, and no re-running this executable
when noise conditions change. The reference PSD is only used
internally, once, to numerically solve for those expansion
coefficients; the result is independent of that choice to the same
tolerance as the 5-harmonic reconstruction fidelity itself (typically
~1e-7 or better — this is checked, not assumed: see the correctness
notes in §5).

**Grid resolution**: 24×24 in `(theta_jn, alpha0)` (default) was chosen
based on convergence testing — an 8×8 grid can underestimate the true
marginalized evidence by several percent to an order of magnitude in
the worst case (a real signal's power redistributes across harmonics
fast enough as it sweeps through frequency that a coarse grid can miss
the peak entirely); 24×24 gets the worst case observed down to ~1%.
This has not been re-validated against the corrected harmonic
generation (§5) at full 5-harmonic resolution, so treat it as a
reasonable starting point rather than a final answer if you have time
to redo that check.

**Cost and storage**: generating the grid is the expensive step (LAL
waveform generation dominates, and cost scales with both the grid
resolution and the number of frequency bins, i.e. with segment length,
via `--delta-f`). Benchmarked at a coarser `--delta-f` than a real
search would use, 24×24 took ~7s/template; at a realistic segment
length (finer `--delta-f`, e.g. the `1/256` Hz used in the example
above) expect substantially more — measure it yourself on a few
templates with `--start-index 0 --end-index 5` before committing to a
full-bank run, rather than trusting a number measured under different
conditions. The output file's size is dominated by the 5 cached raw
harmonics per template (their length is set by
`--sample-rate`/`--delta-f`), not by the grid resolution. For a full
~500k-template bank this is a non-trivial amount of compute and disk,
so you will want to parallelize it — see below.

**Parallelizing across jobs**: `--num-jobs N --job-index I` (`I` from
`0` to `N-1`) splits the bank into `N` contiguous, near-equal chunks and
processes only chunk `I`, writing one output file per job:

```
pycbc_make_tha_marginalization_grid \
    --bank-file BANK.hdf --output GRID_${I}.hdf \
    --sample-rate 2048 --delta-f 0.00390625 --low-frequency-cutoff 20 \
    --num-jobs 100 --job-index ${I} \
    ...  # same options otherwise, identical across every job
```

This is the intended way to run this as an HTCondor/Pegasus-style job
array (one job per `${I}` in `0..N-1`); every job needs the exact same
options apart from `--job-index`, since §4's "must exactly match"
requirement applies across jobs too (mismatched `--interp` or
`--n-theta`/`--n-alpha` between jobs building "the same" grid file will
be caught by `pycbc_combine_tha_marginalization_grid` below, but is
still a mistake to avoid). If you need finer manual control instead
(e.g. uneven chunks), `--start-index`/`--end-index` remain available
and are mutually exclusive with `--num-jobs`/`--job-index`.

You do **not** need to combine the `N` output files before running a
search: `pycbc_inspiral_tha --marginalized-grid-file` accepts multiple
files directly (`--marginalized-grid-file GRID_0.hdf GRID_1.hdf ...`)
and merges them internally. The format is keyed by `template_hash`,
robust to bank thinning/reordering, so partial files are fine as long
as their union covers every template actually being searched. Combining
is purely a convenience (fewer files to track/archive/distribute); if
you want it anyway, `pycbc_combine_tha_marginalization_grid` merges any
number of these files into one:

```
pycbc_combine_tha_marginalization_grid \
    --input-files GRID_0.hdf GRID_1.hdf ... GRID_99.hdf \
    --output GRID_combined.hdf
```

It checks that every input file was built with identical generation
options (same `--sample-rate`/`--delta-f`/`--low-frequency-cutoff`/
`--interp`/`--n-theta`/`--n-alpha`/`--reference-psd-model` — i.e. that
they really are slices of "the same" grid, not accidentally different
configurations) and raises a clear error naming the mismatching
option(s) if not, and warns (without failing) if the same
`template_hash` shows up in more than one input file, which usually
means the `--start-index`/`--end-index` or `--job-index` ranges used to
build them overlapped by mistake.

## 5. Correctness notes worth knowing before extending this

A few non-obvious things were found and fixed while building this that
are worth being aware of if you touch this code:

* **`reverse_flag`**: `bank.py`'s Gram-Schmidt orthogonalization leaves
  the *first* harmonic in the list untouched and orthogonalizes the
  rest against it in sequence, so `reverse_flag` changes which specific
  orthonormal filters come out, not just their order/labeling. This is
  handled (`tha_marg_grid.get_raw_harmonics` reorders before generating
  `A`/`B`, matching `bank.py`'s convention exactly), but it's a subtle
  point if this code is ever refactored: don't assume the 5-harmonic
  basis is invariant under `reverse_flag`.

* **The cyclic time shift**: `bank.py`'s `gen_harmonics_comp` applies a
  `cyclic_time_shift` to correct for Phenom's non-standard FD time
  convention when building `h1..h5`. The `(theta_jn, alpha0)` grid
  points are generated via a separate call (`gen_hp_hc`) that does not
  go through `gen_harmonics_comp`, so it needs the *same* correction
  applied by hand (`compute_raw_and_ab` does this). Without it, the
  grid points and `h1..h5` end up in inconsistent time/phase
  conventions — the symptom is subtle (each harmonic's *magnitude* is
  still correct, only the *relative phase* between harmonics is wrong,
  so the bug does not show up in "does this look like the right order
  of magnitude" checks, only in an actual reconstruction-fidelity
  check). If you ever add another way of generating grid-point
  waveforms, re-run a reconstruction check (see next point) before
  trusting it.

* **Reconstruction fidelity is the load-bearing correctness check for
  this whole approach**: at every `(theta_jn, alpha0)` grid point, the
  actual generated `h+`/`h×` should be reproducible from the cached raw
  harmonics via the `A`/`B` coefficients to a tiny residual (in testing,
  consistently ~1e-7 or better, often ~1e-15). If you change anything
  in the generation path, check this first — it is a much more
  sensitive check than looking at final `marg_lnl` values, which can
  look "plausible" even when the underlying reconstruction is
  completely broken (a broken reconstruction was observed to still
  produce small-but-nonzero, superficially reasonable-looking `marg_lnl`
  values for weak/off-target templates, and only became obviously wrong
  when checked against a known-correct reference).

* **`num_comps` 4/5 harmonic generation**: needs the fix from
  icg-gravwaves/pycbc PR #31 (`whiten_and_normalize_four`/`_five`) to be
  present. Without it, `bank.py` raises `NotImplementedError` for
  `num_comps` 4 or 5 templates. This is unrelated to the marginalization
  work itself but the whole point of this statistic is most relevant
  for the strongly-precessing (`num_comps` 4-5) templates, so make sure
  you're on a branch with this fix.

* **`bank.py`'s `interp=True` shortcut** (in `compute_waveform_five_comps`:
  generate at coarse `df`, then upsample via
  `pycbc.filter.interpolate_complex_frequency`) was investigated in
  detail, since it is what the real search's filtering
  (`get_whitened_normalized_comps`) always uses (hardcoded, no CLI
  override) but this precompute pipeline defaults to the direct,
  non-shortcut `interp=False` path instead (see `--interp` in §4). Two
  things were checked directly (not assumed):

  1. Comparing `bank.py`'s own `compute_waveform_five_comps(interp=True)`
     against `compute_waveform_five_comps(interp=False)` for the same
     template: the 5 harmonics agree in norm to ~1e-7 and in normalized
     overlap to ~1e-6..1e-8. An earlier, since-superseded measurement
     in this project's history had claimed a large (~500-1000x, non-
     uniform across harmonics) discrepancy here; that measurement
     turned out to be confounded by the missing-cyclic-time-shift bug
     described above (which affected grid-point generation, not the
     raw harmonics), not a real property of `interp=True` itself. With
     that bug fixed, `interp=True` and `interp=False` build essentially
     the same `h1..h5` basis.
  2. Using `interp=True` *consistently* for both the raw harmonics and
     the `(theta_jn, alpha0)` grid-point generation (implemented as
     `tha_marg_grid.get_interpolated_hp_hc`, mirroring `bank.py`'s
     `get_interpolated_harmonic_comp` but applied to `h+`/`h×`
     separately) gives `hp_grid = A . raws` reconstruction fidelity of
     only ~1e-4 relative, not the ~1e-14 `interp=False` achieves. The
     reason is a genuine, small numerical property of
     `interpolate_complex_frequency`, not a bug in this pipeline's
     code: it round-trips through a *real*-valued time series
     (`ifft(complex_series, real_time_series)`), which makes it linear
     over real scalars but not over complex ones. `h1..h5`'s
     construction applies the complex phase rotation `exp(2j·dphi)`
     *before* this interpolation step (inside `gen_harmonics_comp`,
     which `get_interpolated_harmonic_comp` calls at the coarse `df`),
     while the grid-point `h+`/`h×` have no such rotation applied
     before interpolation — so the two paths, while each internally
     well-defined, don't compose under exact complex-linearity, leaving
     a small residual. This is a property of `interp=True` as currently
     implemented in `bank.py` generally, not something introduced by
     this precompute pipeline.

  Despite that ~1e-4-level internal residual, an actual end-to-end test
  (build a grid with `--interp`, run `pycbc_inspiral_tha` against an
  injection, compare `marg_lnl` to the same run with a default
  (`interp=False`) grid) showed agreement to ~1e-6 relative precision —
  i.e. `--interp` is a safe, usable option in practice, just not
  perfectly exact internally. `interp=False` remains the default since
  it's exact to machine precision at negligible extra cost for
  reasonably-sized banks.

* **Weakly-precessing templates (low `num_comps`) can have one or more
  of the 5 raw harmonics carry ~zero power** under a given PSD — this is
  exactly why the search truncates them via `num_comps` in the first
  place, but this precompute pipeline always builds and uses the full
  5-harmonic basis regardless. `pycbc.filter.tha_marginalize`'s
  Gram-Schmidt derivation handles this explicitly (treats near-zero-norm
  harmonics as exactly decoupled rather than dividing by zero); if you
  touch that code, keep a low-`num_comps` template in your test set,
  it's the case that exposes numerical edge cases the strongly-
  precessing templates don't.

## 6. Testing scripts

The exploratory/validation scripts used while developing this
(`scripts/*.py` in the parallel `Prec_Marginalization_Test` working
area, not inside the `pycbc` checkout) are not needed to build the grid
file or run a search — everything required for that lives inside
`pycbc` as described above. They're there if useful for further
investigation (e.g. the grid-resolution convergence study, the
`check_phi0_factorization.py` check referenced in §2, the
phase-consistency checks, the injection-recovery tests), but are not
part of the production path.
