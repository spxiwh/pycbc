"""
Runtime support for the 5-harmonic precessing-search marginalized-SNR
statistic (marginalizing over sky-orientation-like nuisance parameters
theta_jn, alpha0, psi, and analytically over distance and phase).

This module intentionally has minimal dependencies (numpy/scipy only)
so it can be imported directly by pycbc_inspiral_tha without pulling in
lalsimulation or the offline bank-time precompute machinery (which lives
in a separate script, since it needs to generate waveforms and is far
too slow to run inline during filtering).

The per-template geometry (grid projection coefficients ``M``, fixed
per-grid-point powers ``hh``, and shared log-prior-weights
``ln_weight``) is computed once offline (see
scripts/build_marg_grid_file.py in the THA precession-marginalization
work) and loaded here via ``load_grid_file``.
"""
from pathlib import Path
import numpy as np
import scipy.special
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
from scipy.special import logsumexp
import h5py

D_LUMINOSITY_MAX = 1.5e4  # Mpc
CACHE_DIR = Path(__file__).parent / '.tha_lookup_table_cache'


def euclidean_distance_prior(d_luminosity):
    """Distance prior uniform in luminosity volume (Mpc^3)."""
    return 4 * np.pi * d_luminosity**2


class LookupTable:
    """
    Marginalize the likelihood over distance via a cached interpolation
    table. Ported from cogwheel's
    cogwheel.likelihood.marginalization.lookup_table.LookupTable,
    stripped of the utils.JSONMixin/numba/IPython dependencies pulled in
    by importing the full cogwheel package, and of the 'comoving'
    distance prior (not needed here).
    """
    REFERENCE_DISTANCE = 1.  # Mpc, luminosity distance at which h is defined.
    _Z0 = 10.
    _SIGMAS = 10.

    def __init__(self, d_luminosity_max=D_LUMINOSITY_MAX, shape=(256, 128)):
        self.d_luminosity_prior = euclidean_distance_prior
        self.d_luminosity_max = d_luminosity_max
        self.shape = shape

        self._inverse_volume = 1 / quad(self.d_luminosity_prior,
                                         0, self.d_luminosity_max)[0]

        x_arr = np.linspace(-self._SIGMAS, 0, shape[0])
        y_arr = np.linspace(self._compactify(-self._SIGMAS / self._Z0),
                             1 - 1e-8, shape[1])
        x_grid, y_grid = np.meshgrid(x_arr, y_arr, indexing='ij')

        dh_grid, hh_grid = self._get_dh_hh(x_grid, y_grid)
        table = self._get_table(dh_grid, hh_grid)
        self._interpolated_table = RectBivariateSpline(x_arr, y_arr, table)

    def _cache_key(self):
        return (f"{self.__class__.__name__}_{self.d_luminosity_max}_"
                f"{self.shape[0]}x{self.shape[1]}")

    def _get_table(self, dh_grid, hh_grid):
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / f"{self._cache_key()}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        table = np.vectorize(self._function)(dh_grid, hh_grid)
        np.save(cache_file, table)
        return table

    def __call__(self, d_h, h_h):
        """log(evidence) - d_h**2 / h_h / 2, evidence marginalized over distance."""
        return self._interpolated_table(*self._get_x_y(d_h, h_h), grid=False)[()]

    def _get_distance_bounds(self, d_h, h_h, sigmas=5.):
        u_peak = d_h / (self.REFERENCE_DISTANCE * h_h)
        delta_u = sigmas / np.sqrt(h_h)
        return np.array([self.REFERENCE_DISTANCE / (u_peak + delta_u),
                          self.REFERENCE_DISTANCE / (u_peak - delta_u)])

    def lnlike_marginalized(self, d_h, h_h):
        """d_h, h_h: inner products (d|h), (h|h) at REFERENCE_DISTANCE (scalars/arrays)."""
        return self(d_h, h_h) + d_h**2 / h_h / 2

    def _function(self, d_h, h_h):
        return np.log(quad(self._function_integrand, 0, self.d_luminosity_max,
                            args=(d_h, h_h),
                            points=self._get_distance_bounds(d_h, h_h))[0]
                       + 1e-100)

    def _function_integrand(self, d_luminosity, d_h, h_h):
        norm_h = np.sqrt(h_h)
        return (self.d_luminosity_prior(d_luminosity) * self._inverse_volume
                * np.exp(-(norm_h * self.REFERENCE_DISTANCE / d_luminosity
                           - d_h / norm_h)**2 / 2))

    def _get_x_y(self, d_h, h_h):
        norm_h = np.sqrt(h_h)
        overlap = d_h / norm_h
        x = np.log(norm_h / (self.d_luminosity_max
                              * (self._SIGMAS + np.abs(overlap))))
        y = self._compactify(overlap / self._Z0)
        return x, y

    def _get_dh_hh(self, x, y):
        overlap = self._uncompactify(y) * self._Z0
        norm_h = (np.exp(x) * self.d_luminosity_max
                  * (self._SIGMAS + np.abs(overlap)))
        d_h = overlap * norm_h
        h_h = norm_h**2
        return d_h, h_h

    @staticmethod
    def _compactify(value):
        return value / (1 + np.abs(value))

    @staticmethod
    def _uncompactify(value):
        return value / (1 - np.abs(value))


class LookupTableMarginalizedPhase22(LookupTable):
    """
    Like LookupTable but additionally marginalizes analytically over
    orbital phase for |m|=2-dominated radiation. d_h is the ABSOLUTE
    VALUE of the complex (d|h).
    """
    def _function_integrand(self, d_luminosity, d_h, h_h):
        return (super()._function_integrand(d_luminosity, d_h, h_h)
                * scipy.special.i0e(d_h * self.REFERENCE_DISTANCE / d_luminosity))


_LOOKUP_TABLE = None


def get_lookup_table():
    """Lazily construct (and cache) the module-level LookupTableMarginalizedPhase22."""
    global _LOOKUP_TABLE
    if _LOOKUP_TABLE is None:
        _LOOKUP_TABLE = LookupTableMarginalizedPhase22()
    return _LOOKUP_TABLE


def marginalize_segment(M, hh, ln_weight, z, lookup_table=None):
    """
    Compute the log-likelihood marginalized over (theta_jn, alpha0, psi,
    distance, phase) for a chunk of per-harmonic complex SNR values.

    Parameters
    ----------
    M : (n_grid, 5) complex array
        Per-grid-point reconstruction coefficients (bank-time precomputed).
    hh : (n_grid,) float array
        Per-grid-point (h|h), independent of the data (bank-time precomputed).
    ln_weight : (n_grid,) float array
        Log of the (theta_jn, alpha0, psi) quadrature weights (shared
        across templates for a fixed grid).
    z : (5,) or (5, n_time) complex array
        snr_comp_1..5 at the sample(s) to evaluate.
    lookup_table : LookupTableMarginalizedPhase22, optional
        Defaults to the shared module-level instance.

    Returns
    -------
    lnl_marg : float or (n_time,) float array
    """
    lookup_table = lookup_table or get_lookup_table()
    z = np.asarray(z)
    scalar_input = (z.ndim == 1)
    if scalar_input:
        z = z[:, None]

    dh = M @ z  # (n_grid, n_time)
    lnl = lookup_table.lnlike_marginalized(np.abs(dh), hh[:, None])
    lnl_marg = logsumexp(ln_weight[:, None] + lnl, axis=0)

    return lnl_marg[0] if scalar_input else lnl_marg


def load_grid_file(path):
    """
    Load an auxiliary per-template marginalization-grid file (as written
    by scripts/build_marg_grid_file.py).

    Returns
    -------
    grid_by_hash : dict
        Maps int(template_hash) -> (M, hh) pair, ready to pass to
        ``marginalize_segment`` along with the shared ``ln_weight``.
    ln_weight : (n_grid,) float array
        Shared across all templates in the file.
    """
    grid_by_hash = {}
    with h5py.File(path, 'r') as f:
        ln_weight = f['ln_weight'][:]
        template_hash = f['template_hash'][:]
        M_all = f['M'][:]
        hh_all = f['hh'][:]
        for i, h in enumerate(template_hash):
            grid_by_hash[int(h)] = (M_all[i], hh_all[i])
    return grid_by_hash, ln_weight
