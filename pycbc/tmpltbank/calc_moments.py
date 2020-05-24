# Copyright (C) 2013 Ian W. Harry
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from __future__ import division
import numpy
try:
    from collections import UserDict
except ImportError:
    # python2
    from UserDict import UserDict
from six.moves import range
from pycbc.tmpltbank.lambda_mapping import generate_mapping

class MomentsDict(UserDict):
    """
    Implements a dictionary of noise moments computed only when needed.

    This class provides a dictionary of noise moments. Moments are only
    computed if needed (and cached for future calls).
    """
    def __init__(self, inp_dict, metric_params=None, vary_fmax=False,
                 vary_density=None, **kwargs):
        # The python2 UserDict is a bit rubbish! It's not a new-style class
        # so no super can be done. Move to super when python2 is deprecated.
        #super(MomentsDict, self).__init__(inp_dict, **kwargs)
        UserDict.__init__(self, inp_dict, **kwargs)
        if metric_params is None:
            err_msg = "metric_params keyword argument must be provided."
            raise ValueError(err_msg)

        self.metric_params = metric_params
        
        psd_amp = metric_params.psd.data
        psd_f = numpy.arange(len(psd_amp), dtype=float) * metric_params.deltaF
        new_f, new_amp = interpolate_psd(psd_f, psd_amp, metric_params.deltaF)
        self.new_f = new_f
        self.new_amp = new_amp

        # We need to compute the (7,0) moment as that is a normalization
        # factor
        funct = lambda x,f0: 1
        norm = calculate_moment(new_f, new_amp, self.metric_params.fLow,
                                self.metric_params.fUpper,
                                self.metric_params.f0,
                                funct, vary_fmax=vary_fmax,
                                vary_density=vary_density)
        self.norm = norm
        self.data[(7,0)] = 1



    def __setitem__(self, key, value):
        err_msg = "MomentsDict does not support setting values directly"
        raise ValueError(err_msg) 

    def __delitem__(self, key):
        err_msg = "MomentsDict does not support deleting values directly"
        raise ValueError(err_msg)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            assert isinstance(key, tuple)
            assert len(key) == 2
            order = key[0]
            logorder = key[1]
            funct = lambda x,f0: (numpy.log((x*f0)**(1./3.)))**logorder \
                * x**((-order+7)/3.)
            value = calculate_moment(new_f, new_amp, self.metric_params.fLow,
                                self.metric_params.fUpper,
                                self.metric_params.f0,
                                funct, vary_fmax=vary_fmax,
                                vary_density=vary_density)
            self.data[key] = value
            return value




def determine_eigen_directions(metricParams, preserveMoments=False,
                               vary_fmax=False, vary_density=None):
    """
    This function will calculate the coordinate transfomations that are needed
    to rotate from a coordinate system described by the various Lambda
    components in the frequency expansion, to a coordinate system where the
    metric is Cartesian.

    Parameters
    -----------
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric.
    preserveMoments : boolean, optional (default False)
        Currently only used for debugging.
        If this is given then if the moments structure is already set
        within metricParams then they will not be recalculated.
    vary_fmax : boolean, optional (default False)
        If set to False the metric and rotations are calculated once, for the
        full range of frequency [f_low,f_upper).
        If set to True the metric and rotations are calculated multiple times,
        for frequency ranges [f_low,f_low + i*vary_density), where i starts at
        1 and runs up until f_low + (i+1)*vary_density > f_upper.
        Thus values greater than f_upper are *not* computed.
        The calculation for the full range [f_low,f_upper) is also done.
    vary_density : float, optional
        If vary_fmax is True, this will be used in computing the frequency
        ranges as described for vary_fmax.

    Returns
    --------
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric.
        **THIS FUNCTION ONLY RETURNS THE CLASS**
        The following will be **added** to this structure
    metricParams.evals : Dictionary of numpy.array
        Each entry in the dictionary corresponds to the different frequency
        ranges described in vary_fmax. If vary_fmax = False, the only entry
        will be f_upper, this corresponds to integrals in [f_low,f_upper). This
        entry is always present. Each other entry will use floats as keys to
        the dictionary. These floats give the upper frequency cutoff when it is
        varying.
        Each numpy.array contains the eigenvalues which, with the eigenvectors
        in evecs, are needed to rotate the
        coordinate system to one in which the metric is the identity matrix.
    metricParams.evecs : Dictionary of numpy.matrix
        Each entry in the dictionary is as described under evals.
        Each numpy.matrix contains the eigenvectors which, with the eigenvalues
        in evals, are needed to rotate the
        coordinate system to one in which the metric is the identity matrix.
    metricParams.metric : Dictionary of numpy.matrix
        Each entry in the dictionary is as described under evals.
        Each numpy.matrix contains the metric of the parameter space in the
        Lambda_i coordinate system.
    metricParams.moments : Moments structure
        See the structure documentation for a description of this. This
        contains the result of all the integrals used in computing the metrics
        above. It can be used for the ethinca components calculation, or other
        similar calculations.
    """

    evals = {}
    evecs = {}
    metric = {}
    unmax_metric = {}

    # First step is to get the moments needed to calculate the metric
    if not (metricParams.moments and preserveMoments):
        metricParams.moments = MomentsDict({}, metric_params=metricParams,
                                           vary_fmax=vary_fmax,
                                           vary_density=vary_density)


    # What values are going to be in the moments
    # We start looping over every item in the list of metrics
    for term_freq in metricParams.moments[(7, 0)].keys():
        mapping = generate_mapping(metricParams.pnOrder)

        # Calculate the metric
        gs, unmax_metric_curr = calculate_metric(metricParams.moments, mapping,
                                                 term_freq)
        metric[term_freq] = numpy.matrix(gs)
        unmax_metric[term_freq] = unmax_metric_curr

        # And the eigenvalues
        evals[term_freq],evecs[term_freq] = numpy.linalg.eig(gs)

        # Numerical error can lead to small negative eigenvalues.
        for i in range(len(evals[term_freq])):
            if evals[term_freq][i] < 0:
                # Due to numerical imprecision the very small eigenvalues can
                # be negative. Make these positive.
                evals[term_freq][i] = -evals[term_freq][i]
            if evecs[term_freq][i,i] < 0:
                # We demand a convention that all diagonal terms in the matrix
                # of eigenvalues are positive.
                # This is done to help visualization of the spaces (increasing
                # mchirp always goes the same way)
                evecs[term_freq][:,i] = - evecs[term_freq][:,i]

    metricParams.evals = evals
    metricParams.evecs = evecs
    metricParams.metric = metric
    metricParams.time_unprojected_metric = unmax_metric

    return metricParams

def interpolate_psd(psd_f, psd_amp, deltaF):
    """
    Function to interpolate a PSD to a different value of deltaF. Uses linear
    interpolation.

    Parameters
    ----------
    psd_f : numpy.array or list or similar
        List of the frequencies contained within the PSD.
    psd_amp : numpy.array or list or similar
        List of the PSD values at the frequencies in psd_f.
    deltaF : float
        Value of deltaF to interpolate the PSD to.

    Returns
    --------
    new_psd_f : numpy.array
       Array of the frequencies contained within the interpolated PSD
    new_psd_amp : numpy.array
       Array of the interpolated PSD values at the frequencies in new_psd_f.
    """
    # In some cases this will be a no-op. I thought about removing this, but
    # this function can take unequally sampled PSDs and it is difficult to
    # check for this. As this function runs quickly anyway (compared to the
    # moment calculation) I decided to always interpolate.

    new_psd_f = []
    new_psd_amp = []
    fcurr = psd_f[0]

    for i in range(len(psd_f) - 1):
        f_low = psd_f[i]
        f_high = psd_f[i+1]
        amp_low = psd_amp[i]
        amp_high = psd_amp[i+1]
        while(1):
            if fcurr > f_high:
                break
            new_psd_f.append(fcurr)
            gradient = (amp_high - amp_low) / (f_high - f_low)
            fDiff = fcurr - f_low
            new_psd_amp.append(amp_low + fDiff * gradient)
            fcurr = fcurr + deltaF
    return numpy.asarray(new_psd_f), numpy.asarray(new_psd_amp)


def calculate_moment(psd_f, psd_amp, fmin, fmax, f0, funct,
                     norm=None, vary_fmax=False, vary_density=None):
    """
    Function for calculating one of the integrals used to construct a template
    bank placement metric. The integral calculated will be

    \int funct(x) * (psd_x)**(-7./3.) * delta_x / PSD(x)

    where x = f / f0. The lower frequency cutoff is given by fmin, see
    the parameters below for details on how the upper frequency cutoff is
    chosen

    Parameters
    -----------
    psd_f : numpy.array
       numpy array holding the set of evenly spaced frequencies used in the PSD
    psd_amp : numpy.array
       numpy array holding the PSD values corresponding to the psd_f
       frequencies
    fmin : float
        The lower frequency cutoff used in the calculation of the integrals
        used to obtain the metric.
    fmax : float
        The upper frequency cutoff used in the calculation of the integrals
        used to obtain the metric. This can be varied (see the vary_fmax
        option below).
    f0 : float
        This is an arbitrary scaling factor introduced to avoid the potential
        for numerical overflow when calculating this. Generally the default
        value (70) is safe here. **IMPORTANT, if you want to calculate the
        ethinca metric components later this MUST be set equal to f_low.**
    funct : Lambda function
        The function to use when computing the integral as described above.
    norm : Dictionary of floats
        If given then moment[f_cutoff] will be divided by norm[f_cutoff]
    vary_fmax : boolean, optional (default False)
        If set to False the metric and rotations are calculated once, for the
        full range of frequency [f_low,f_upper).
        If set to True the metric and rotations are calculated multiple times,
        for frequency ranges [f_low,f_low + i*vary_density), where i starts at
        1 and runs up until f_low + (i+1)*vary_density > f_upper.
        Thus values greater than f_upper are *not* computed.
        The calculation for the full range [f_low,f_upper) is also done.
    vary_density : float, optional
        If vary_fmax is True, this will be used in computing the frequency
        ranges as described for vary_fmax.

    Returns
    --------
    moment : Dictionary of floats
        moment[f_cutoff] will store the value of the moment at the frequency
        cutoff given by f_cutoff.
    """

    # Must ensure deltaF in psd_f is constant
    psd_x = psd_f / f0
    deltax = psd_x[1] - psd_x[0]

    mask = numpy.logical_and(psd_f > fmin, psd_f < fmax)
    psdf_red = psd_f[mask]
    comps_red = psd_x[mask] ** (-7./3.) * funct(psd_x[mask], f0) * deltax / \
                psd_amp[mask]
    moment = {}
    moment[fmax] = comps_red.sum()
    if norm:
        moment[fmax] = moment[fmax] / norm[fmax]
    if vary_fmax:
        for t_fmax in numpy.arange(fmin + vary_density, fmax, vary_density):
            moment[t_fmax] = comps_red[psdf_red < t_fmax].sum()
            if norm:
                moment[t_fmax] = moment[t_fmax] / norm[t_fmax]
    return moment

def calculate_metric(metric_moments, mapping, term_freq):
    """
    This function will take the various integrals calculated by get_moments and
    convert this into a metric for the appropriate parameter space.

    Returns
    --------
    metric : numpy.matrix
        The resulting metric.
    """

    # How many dimensions in the parameter space?
    maxLen = len(mapping.keys())

    metric = numpy.matrix(numpy.zeros(shape=(maxLen,maxLen),dtype=float))
    unmax_metric = numpy.matrix(numpy.zeros(shape=(maxLen+1,maxLen+1),
                                                                  dtype=float))

    for term_1 in mapping:
        for term_2 in mapping:
            calculate_metric_comp(metric, unmax_metric, term_1, term_2,
                                  metric_moments, term_freq)
    return metric, unmax_metric

def identify_orders_from_string(term_str):
    """
    Map from order string to numerical order.

    This function takes an order string in the format LambdaN or LogLambdaN
    or similar and returns a tuple corresponding to that terms order. The
    first value corresponds to the N the second value corresponds to the
    power of the Log term.
    """
    log_order = 0
    # Strip off Log terms
    while term_str.startswith('Log'):
        log_order += 1
        term_str = term_str[3:]
    # Now remove the leading "Lambda"
    term_str = term_str.replace('Lambda', '')
    pn_order = int(term_str)
    return (pn_order, log_order)


def calculate_metric_comp(gs, unmax_metric, term_1, term_2,
                          metric_moments, term_freq):
    """
    Used to compute part of the metric. Only call this from within
    calculate_metric(). Please see the documentation for that function.
    """
    # Time term in unmax_metric. Note that these terms are recomputed a bunch
    # of time, but this cost is insignificant compared to computing the moments
    unmax_metric[-1,-1] = (Js[1] - Js[4]*Js[4])

    # Identify input details
    term1_orders = identify_orders_from_string(term_1)
    term2_orders = identify_orders_from_string(term_2)

    # And moments needed
    moment_1 = metric_moments[(17 - term1_orders[0] - term2_orders[0],
                               term1_orders[1] + term2_orders[1])][term_freq]
    moment_2 = metric_moments[(12 - term1_orders[0],
                               term1_orders[1])][term_freq]
    moment_3 = metric_moments[(12 - term2_orders[0], 
                               term2_orders[1])][term_freq]
    moment_4 = metric_moments[(9 - term1_orders[0], 
                               term1_orders[1])][term_freq]
    moment_5 = metric_moments[(9 - term2_orders[0],
                               term2_orders[1])][term_freq]
    moment_6 = metric_moments[(4,0)][term_freq]
    moment_7 = metric_moments[(1,0)][term_freq]

    # And gamma terms
    gammaij = moment_1 - moment_2*moment_3
    gamma0i = moment_4 - moment_6*moment_2
    gamma0j = moment_5 - moment_6*moment_3

    # And then metric terms
    gs[mapping[term1], mapping[term2]] = \
        0.5 * (gammaij - gamma0i*gamma0j/(moment_7 - moment_6*moment_6))
    unmax_metric[mapping[term1], -1] = gamma0i
    unmax_metric[-1, mapping[term1]] = gamma0i
    unmax_metric[mapping[term2], -1] = gamma0j
    unmax_metric[-1, mapping[term2]] = gamma0j 
    unmax_metric[mapping[term1], mapping[term2]] = gammaij
