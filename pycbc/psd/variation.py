""" PSD Variation """

import numpy
from numpy.fft import rfft, irfft
import scipy.signal as sig
from scipy.interpolate import interp1d

import pycbc.psd
from pycbc.psd import interpolate
from pycbc.types import TimeSeries
from pycbc.filter import make_frequency_series, fir_zero_filter
from pycbc.filter.resample import cached_firwin
from pycbc.vetoes import power_chisq_bins


def create_full_filt(freqs, filt, plong, srate, psd_duration, low_freq, high_freq):
    """Create a filter to convolve with strain data to find PSD variation.

    Parameters
    ----------
    freqs : numpy.ndarray
        Array of sample frequencies of the PSD.
    filt : numpy.ndarray
        A bandpass filter.
    plong : numpy.ndarray
        The estimated PSD.
    srate : float
        The sample rate of the data.
    psd_duration : float
        The duration of the estimated PSD.

    Returns
    -------
    full_filt : numpy.ndarray
        The full filter used to calculate PSD variation.
    """

    # Make the weighting filter - bandpass, which weight by f^-7/6,
    # and whiten. The normalization is chosen so that the variance
    # will be one if this filter is applied to white noise which
    # already has a variance of one.
    fweight = freqs ** (-7./6.) * filt / numpy.sqrt(plong)
    fweight[0] = 0
    # Need to impose frequency limits here to avoid issues if the PSD goes to
    # 0 (e.g. at very low frequencies), which can then dominate the final
    # filter produced.
    # Allow some buffer here to avoid abrupt terminations
    # FIXME: This is a bit hacky, better HP filters would avoid this
    fweight[freqs < low_freq*0.9] = 0
    fweight[freqs > high_freq*1.1] = 0
    norm = (sum(abs(fweight) ** 2) / (len(fweight) - 1.)) ** -0.5
    fweight = norm * fweight
    fwhiten = numpy.sqrt(2. / srate) / numpy.sqrt(plong)
    fwhiten[0] = 0.
    full_filt = sig.windows.hann(int(psd_duration * srate)) * numpy.roll(
        irfft(fwhiten * fweight), int(psd_duration / 2) * srate)

    return full_filt


def build_band_kernels(fbins, sample_rate, kernel_duration=2.0):
    """ Build a short, band-selective FIR kernel for each PSD-variation
    frequency bin.

    These kernels are the time-domain impulse response of a bandpass
    selecting just one psdvar frequency bin. They depend only on the
    (fixed) bin edges and sample rate, never on a template or on the
    strain content, so they can be computed once per PSD update and
    reused for every template and every noisy time window: applying one
    of them to a matched-filter SNR time series is mathematically the
    same operation as zeroing out `corr` outside that bin and inverse
    Fourier transforming, but done as a short local convolution rather
    than a transform over the full analysis segment.

    `kernel_duration` trades off frequency resolution (how cleanly the
    kernel isolates its bin from its neighbours) against the size, and
    therefore cost, of the local correction it can later be used for.

    Parameters
    ----------
    fbins : array of float
        Edges of the psdvar frequency bins, in Hz (length nbins + 1).
    sample_rate : float
        Sample rate of the strain/SNR data the kernels will be applied to.
    kernel_duration : float, optional
        Duration, in seconds, of each FIR kernel.

    Returns
    -------
    kernels : list of numpy.ndarray
        One real-valued FIR kernel per bin (length nbins).
    """
    ntaps = int(kernel_duration * sample_rate)
    # firwin needs an odd number of taps for a Type I (symmetric, integer
    # group delay) linear-phase filter.
    ntaps |= 1
    kernels = []
    for f_lo, f_hi in zip(fbins[:-1], fbins[1:]):
        # cached_firwin is functools.lru_cache-wrapped, so its arguments
        # must be hashable -- a tuple, not a list, for the band edges.
        taps = cached_firwin(ntaps, (float(f_lo), float(f_hi)), pass_zero=False,
                              window='hann', fs=sample_rate)
        kernels.append(taps)
    return kernels


def mean_square(data, delta_t, srate, short_stride, stride, glitch_remover=True):
    """ Calculate mean square of given time series once per stride

    First of all this function calculate the mean square of given time
    series once per short_stride. This is used to find and remove
    outliers due to short glitches. Here an outlier is defined as any
    element which is greater than two times the average of its closest
    neighbours. Every outlier is substituted with the average of the
    corresponding adjacent elements.
    Then, every second the function compute the mean square of the
    smoothed time series, within the stride.

    Parameters
    ----------
    data : numpy.ndarray
    delta_t : float
        Duration of the time series
    srate : int
        Sample rate of the data were it given as a TimeSeries
    short_stride : float
        Stride duration for outlier removal
    stride ; float
        Stride duration

    Returns
    -------
    m_s: List
        Mean square of given time series
    """

    # Calculate mean square of data once per short stride and replace
    # outliers
    short_ms = numpy.mean(data.reshape(-1, int(srate * short_stride)) ** 2,
                          axis=1)
    # Define an array of averages that is used to substitute outliers
    if glitch_remover:
        ave = 0.5 * (short_ms[2:] + short_ms[:-2])
        outliers = short_ms[1:-1] > (2. * ave)
        short_ms[1:-1][outliers] = ave[outliers]

    # Calculate mean square of data every step within a window equal to
    # stride seconds
    m_s = []
    inv_time = int(1. / short_stride)
    for index in range(int(delta_t - stride + 1)):
        m_s.append(numpy.mean(short_ms[inv_time * index:inv_time *
                                       int(index+stride)]))
    return m_s


def calc_filt_psd_variation(strain, segment, short_segment, psd_long_segment,
                            psd_duration, psd_stride, psd_avg_method, low_freq,
                            high_freq, glitch_remover=True):
    """ Calculates time series of PSD variability

    This function first splits the segment up into 512 second chunks. It
    then calculates the PSD over this 512 second. The PSD is used to
    to create a filter that is the composition of three filters:
    1. Bandpass filter between f_low and f_high.
    2. Weighting filter which gives the rough response of a CBC template.
    3. Whitening filter.
    Next it makes the convolution of this filter with the stretch of data.
    This new time series is given to the "mean_square" function, which
    computes the mean square of the timeseries within an 8 seconds window,
    once per second.
    The result, which is the variance of the S/N in that stride for the
    Parseval theorem, is then stored in a timeseries.

    Parameters
    ----------
    strain : TimeSeries
        Input strain time series to estimate PSDs
    segment : {float, 8}
        Duration of the segments for the mean square estimation in seconds.
    short_segment : {float, 0.25}
        Duration of the short segments for the outliers removal.
    psd_long_segment : {float, 512}
        Duration of the long segments for PSD estimation in seconds.
    psd_duration : {float, 8}
        Duration of FFT segments for long term PSD estimation, in seconds.
    psd_stride : {float, 4}
        Separation between FFT segments for long term PSD estimation, in
        seconds.
    psd_avg_method : {string, 'median'}
        Method for averaging PSD estimation segments.
    low_freq : {float, 20}
        Minimum frequency to consider the comparison between PSDs.
    high_freq : {float, 480}
        Maximum frequency to consider the comparison between PSDs.

    Returns
    -------
    psd_var : TimeSeries
        Time series of the variability in the PSD estimation
    """
    # Calculate strain precision
    if strain.precision == 'single':
        fs_dtype = numpy.float32
    elif strain.precision == 'double':
        fs_dtype = numpy.float64

    # Convert start and end times immediately to floats
    start_time = float(strain.start_time)
    end_time = float(strain.end_time)
    srate = int(strain.sample_rate)

    # Fix the step for the PSD estimation and the time to remove at the
    # edge of the time series.
    step = 1.0
    strain_crop = 8.0

    # Find the times of the long segments
    times_long = numpy.arange(start_time, end_time,
                              psd_long_segment - 2 * strain_crop
                              - segment + step)

    # Create a bandpass filter between low_freq and high_freq
    filt = sig.firwin(4 * srate, [low_freq, high_freq], pass_zero=False,
                      window='hann', fs=srate)
    filt.resize(int(psd_duration * srate))
    # Fourier transform the filter and take the absolute value to get
    # rid of the phase.
    filt = abs(rfft(filt))

    psd_var_list = []
    for tlong in times_long:
        # Calculate PSD for long segment
        if tlong + psd_long_segment <= float(end_time):
            astrain = strain.time_slice(tlong, tlong + psd_long_segment)
            plong = pycbc.psd.welch(
                astrain,
                seg_len=int(psd_duration * strain.sample_rate),
                seg_stride=int(psd_stride * strain.sample_rate),
                avg_method=psd_avg_method)
        else:
            astrain = strain.time_slice(tlong, end_time)
            plong = pycbc.psd.welch(
                           strain.time_slice(end_time - psd_long_segment,
                                             end_time),
                           seg_len=int(psd_duration * strain.sample_rate),
                           seg_stride=int(psd_stride * strain.sample_rate),
                           avg_method=psd_avg_method)
        freqs = numpy.array(plong.sample_frequencies, dtype=fs_dtype)
        plong = plong.numpy()

        full_filt = create_full_filt(freqs, filt, plong, srate, psd_duration, low_freq, high_freq)
        # Convolve the filter with long segment of data. Use pycbc's own
        # FFT-based FIR filtering (which internally chunks long inputs)
        # rather than scipy, so this stays on pycbc's fft/scheme machinery.
        wstrain = fir_zero_filter(full_filt, astrain).numpy()
        wstrain = wstrain[int(strain_crop * srate):-int(strain_crop * srate)]
        # compute the mean square of the chunk of data
        delta_t = len(wstrain) * strain.delta_t
        variation = mean_square(wstrain, delta_t, srate, short_segment, segment, glitch_remover=glitch_remover)
        psd_var_list.append(numpy.array(variation, dtype=wstrain.dtype))

    # Package up the time series to return
    psd_var = TimeSeries(numpy.concatenate(psd_var_list), delta_t=step,
                         epoch=start_time + strain_crop + segment)

    return psd_var


def get_psdvar_f_bins(nbins, template, psd, low_freq, high_freq):
    '''
    finds the edges of the chisq frequency bins

    Parameters:
    -----------
    nbins : int
        number of frequency bins
    template : TimeSeries
        merger template
    psd : FrequencySeries
        psd of the strain data
    low_freq : float
        lower bound on frequency of data in Hertz
    high_freq : float
        upper bound on frequency of data in Hertz

    Returns:
    --------
    bins : list
        indices of the frequency bin edges
    fbins : list
        frequency values of the frequency bin edges (Hz)
    '''
    htilde = make_frequency_series(template)
    psd_interp = interpolate(psd, htilde.delta_f)
    bins = power_chisq_bins(htilde,
                            nbins,
                            psd_interp,
                            low_freq,
                            high_freq)
    fbins = bins * template.delta_f
    return bins, fbins


def get_psdvar_freq_dict(data, fbins, segment=8., short_segment=0.25,
                         psd_long_segment=512., psd_duration=8., psd_stride=4.,
                         psd_avg_method='median', glitch_remover=True):
    '''
    makes a dictionary of psd variation across frequencies

    Parameters:
    -----------
    data : TimeSeries
        strain data
    fbins : array
        edges of the frequency bins (Hz)
    segment : float, optional
    short_segment : float, optional
    psd_long_segment : float, optional
    psd_duration : float, optional
    psd_stride : float, optional
    psd_avg_method : string, optional
    glitch_remover : bool, optional

    Returns:
    --------
    var_dict: dict
        the psd variation in each frequency bin per timestamp {timestamp (float): non-stationarity (list) }
    '''
    var_dict_raw = {}
    # calculate psd variation for each frequency bin
    for f in range(len(fbins)-1):
        var = calc_filt_psd_variation(
            data, segment, short_segment, psd_long_segment,
            psd_duration, psd_stride, psd_avg_method,
            low_freq=fbins[f], high_freq=fbins[f+1],
            glitch_remover=glitch_remover
        )
        var_dict_raw[fbins[f]] = var

    # put in format {time: [variations]}
    timestamps = var_dict_raw[fbins[0]].sample_times.numpy()
    var_array = numpy.array([v.numpy() for v in var_dict_raw.values()])  # shape: (F, T)

    var_dict = {
        float(timestamps[i]): var_array[:, i]
        for i in range(len(timestamps))
    }
    return var_dict


class PSDVariation(object):
    @classmethod
    def from_cli(cls, opt, strain, glitch_remover=True, nbins=10):
        is_enabled = hasattr(opt, 'psdvar_segment') and opt.psdvar_segment is not None
        if not is_enabled:
            return None

        import logging
        logging.info("Calculating PSD variation")

        segment = opt.psdvar_segment
        short_segment = opt.psdvar_short_segment
        long_segment = opt.psdvar_long_segment
        psd_duration = opt.psdvar_psd_duration
        psd_stride = opt.psdvar_psd_stride
        avg_method = getattr(opt, 'psd_estimation', 'median')
        low_freq = opt.psdvar_low_freq
        high_freq = opt.psdvar_high_freq

        freq_dep = getattr(opt, 'psdvar_freq_dependent', False)
        threshold = getattr(opt, 'psdvar_threshold', 1.6)

        fbins = None
        if freq_dep:
            if hasattr(opt, 'psdvar_freq_bins') and opt.psdvar_freq_bins:
                fbins = [float(f) for f in opt.psdvar_freq_bins.split(',')]
            else:
                fbins = numpy.geomspace(low_freq, high_freq, nbins + 1)

        obj = cls(strain, freq_dep, fbins, threshold, segment,
                   short_segment, long_segment, psd_duration,
                   psd_stride, avg_method, low_freq, high_freq, glitch_remover)

        # Disable storing PSD variation numbers into events if we're using frequency-dependent
        if not obj.store_in_events:
            import logging
            if freq_dep:
                logging.info("Frequency dependent PSD variation generated. Standard PSD variation output disabled.")
            opt.psdvar_segment = None
            opt.psdvar_short_segment = None

        return obj

    def __init__(self, strain, frequency_dependent, fbins, threshold, segment,
                 short_segment, long_segment, psd_duration,
                 psd_stride, avg_method, low_freq, high_freq, glitch_remover,
                 kernel_duration=2.0):
        self.frequency_dependent = frequency_dependent
        self.fbins = fbins
        self.threshold = threshold
        self.store_in_events = not frequency_dependent and short_segment is not None
        self.bad_windows = {}
        self.bin_kernels = None

        if frequency_dependent:
            self.var_dict = get_psdvar_freq_dict(
                strain, self.fbins, segment=segment, short_segment=short_segment,
                psd_long_segment=long_segment, psd_duration=psd_duration,
                psd_stride=psd_stride, psd_avg_method=avg_method,
                glitch_remover=glitch_remover)
            self.data = None

            # Precompute, once, the small set of (second, bin) pairs that
            # actually exceed the variation threshold. Both the SNR
            # correction (matchedfilter.py) and the chisq correction
            # (pycbc_inspiral) look this up by rounded GPS second instead
            # of rescanning the whole var_dict per template/trigger.
            self.bad_windows = {
                int(round(t)): [(i, v) for i, v in enumerate(vals)
                                if v > threshold]
                for t, vals in self.var_dict.items()
                if any(v > threshold for v in vals)
            }

            # Short, template-independent band-selective FIR kernels, one
            # per frequency bin, used to build a local correction directly
            # on the SNR time series instead of redoing a full-segment IFFT.
            self.bin_kernels = build_band_kernels(
                self.fbins, strain.sample_rate, kernel_duration=kernel_duration)
        else:
            self.var_dict = None
            self.data = calc_filt_psd_variation(
                strain, segment, short_segment, long_segment,
                psd_duration, psd_stride, avg_method, low_freq,
                high_freq, glitch_remover=glitch_remover)

    def items(self):
        if self.frequency_dependent:
            return self.var_dict.items()
        return None


def find_trigger_value(psd_var, idx, start, sample_rate):
    """ Find the PSD variation value at a particular time with the filter
    method. If the time is outside the timeseries bound, 1. is given.

    Parameters
    ----------
    psd_var : TimeSeries
        Time series of the varaibility in the PSD estimation
    idx : numpy.ndarray
        Time indices of the triggers
    start : float
        GPS start time
    sample_rate : float
        Sample rate defined in ini file

    Returns
    -------
    vals : Array
        PSD variation value at a particular time
    """
    # Find gps time of the trigger
    time = start + idx / sample_rate
    # Extract the PSD variation at trigger time through linear
    # interpolation
    if not hasattr(psd_var, 'cached_psd_var_interpolant'):
        psd_var.cached_psd_var_interpolant = \
            interp1d(psd_var.sample_times.numpy(),
                     psd_var.numpy(),
                     fill_value=1.0,
                     bounds_error=False)
    vals = psd_var.cached_psd_var_interpolant(time)

    return vals


def live_create_filter(psd_estimated,
                       psd_duration,
                       sample_rate,
                       low_freq=20,
                       high_freq=480):
    """
    Create a filter to be used in the calculation of the psd variation for the
    PyCBC Live search. This filter combines a bandpass between a lower and
    upper frequency and an estimated signal response so that the variance
    will be 1 when the filter is applied to white noise.

    Within the PyCBC Live search this filter needs to be recreated every time
    the estimated psd is updated and needs to be unique for each detector.

    Parameters
    ----------
    psd_estimated : pycbc.frequencyseries
        The current PyCBC Live PSD: variations are measured relative to this
        estimate.
    psd_duration : float
        The duration of the estimation of the psd, in seconds.
    sample_rate : int
        The sample rate of the strain data being search over.
    low_freq : int (default = 20)
        The lower frequency to apply in the bandpass filter.
    high_freq : int (default = 480)
        The upper frequency to apply in the bandpass filter.

    Returns
    -------
    full_filt : numpy.ndarray
        The complete filter to be convolved with the strain data to
        find the psd variation value.

    """

    # Create a bandpass filter between low_freq and high_freq once
    filt = sig.firwin(4 * sample_rate,
                      [low_freq, high_freq],
                      pass_zero=False,
                      window='hann',
                      fs=sample_rate)
    filt.resize(int(psd_duration * sample_rate))

    # Fourier transform the filter and take the absolute value to get
    #  rid of the phase.
    filt = abs(rfft(filt))

    # Extract the psd frequencies to create a representative filter.
    freqs = numpy.array(psd_estimated.sample_frequencies, dtype=numpy.float32)
    plong = psd_estimated.numpy()
    full_filt = create_full_filt(freqs, filt, plong, sample_rate, psd_duration)

    return full_filt


def live_calc_psd_variation(strain,
                            full_filt,
                            increment,
                            data_trim=2.0,
                            short_stride=0.25):
    """
    Calculate the psd variation in the PyCBC Live search.

    The Live strain data is convolved with the filter to produce a timeseries
    containing the PSD variation values for each sample. The mean square of
    the timeseries is calculated over the short_stride to find outliers caused
    by short duration glitches. Outliers are replaced with the average of
    adjacent elements in the array. This array is then further averaged every
    second to produce the PSD variation timeseries.

    Parameters
    ----------
    strain : pycbc.timeseries
        Live data being searched through by the PyCBC Live search.
    full_filt : numpy.ndarray
        A filter created by `live_create_filter`.
    increment : float
        The number of seconds in each increment in the PyCBC Live search.
    data_trim : float
        The number of seconds to be trimmed from either end of the convolved
        timeseries to prevent artefacts.
    short_stride : float
        The number of seconds to average the PSD variation timeseries over to
        remove the effects of short duration glitches.

    Returns
    -------
    psd_var : pycbc.timeseries
        A timeseries containing the PSD variation values.

    """
    sample_rate = int(strain.sample_rate)

    # Grab the last increments worth of data, plus padding for edge effects.
    astrain = strain.time_slice(strain.end_time - increment - (data_trim * 3),
                                strain.end_time)

    # Convolve the data and the filter to produce the PSD variation timeseries,
    #  then trim the beginning and end of the data to prevent edge effects.
    # Uses pycbc's own FFT-based FIR filtering rather than scipy.
    wstrain = fir_zero_filter(full_filt, astrain).numpy()
    wstrain = wstrain[int(data_trim * sample_rate):-int(data_trim * sample_rate)]

    # Create a PSD variation array by taking the mean square of the PSD
    #  variation timeseries every short_stride
    short_ms = numpy.mean(
        wstrain.reshape(-1, int(sample_rate * short_stride)) ** 2, axis=1)

    # Define an array of averages that is used to substitute outliers
    ave = 0.5 * (short_ms[2:] + short_ms[:-2])
    outliers = short_ms[1:-1] > (2. * ave)
    short_ms[1:-1][outliers] = ave[outliers]

    # Calculate the PSD variation every second by a moving window average
    # containing (1/short_stride) short_ms samples.
    m_s = []
    samples_per_second = 1 / short_stride
    for idx in range(int(len(short_ms) / samples_per_second)):
        start = int(samples_per_second * idx)
        end = int(samples_per_second * (idx + 1))
        m_s.append(numpy.mean(short_ms[start:end]))

    m_s = numpy.array(m_s, dtype=wstrain.dtype)
    psd_var = TimeSeries(m_s,
                         delta_t=1.0,
                         epoch=strain.end_time - increment - (data_trim * 2))

    return psd_var


def live_find_var_value(triggers,
                        psd_var_timeseries):
    """
    Extract the PSD variation values at trigger times by linear interpolation.

    Parameters
    ----------
    triggers : dict
        Dictionary containing input trigger times.
    psd_var_timeseries : pycbc.timeseries
        A timeseries containing the PSD variation value for each second of the
        latest increment in PyCBC Live.

    Returns
    -------
    psd_var_vals : numpy.ndarray
        Array of interpolated PSD variation values at trigger times.
    """

    # Create the interpolator
    interpolator = interp1d(psd_var_timeseries.sample_times.numpy(),
                            psd_var_timeseries.numpy(),
                            fill_value=1.0,
                            bounds_error=False)
    # Evaluate at the trigger times
    psd_var_vals = interpolator(triggers['end_time'])

    return psd_var_vals
