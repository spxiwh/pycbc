# Add copyright stuff

import h5py
from scipy.interpolate import UnivariateSpline

import lal

from pycbc.waveform import seobnrrom_length_in_time

def get_data_from_h5_file(filename, time_series):
    """
    This one is a bit of a Ronseal.
    """
    fp = h5py.file(filename, 'r')
    deg = fp['deg'][()]
    knots = fp['knots'][:]
    coeffs = fp['coeffs'][:]
    indices = fp['indices'][:]
    data = fp['data'][:]
    # Check time_series is valid
    assert(knots[0] < time_series[0])
    assert(knots[-1] > time_series[-1])
    fp.close()
    spline = UnivariateSpline(self.knots, self._data, k=self._deg, s=0)
    spline(time_series, dx=0)
    return spline

def get_hplus_hcross_from_directory(directory, template_params):
    """
    Generate hplus, hcross from a directory, for a given total mass and
    f_lower.
    """
    total_mass = template_params['mtotal']
    flower = template_params['flower']
    delta_t = template_params['delta_t']
    theta = template_params['inclination'] # Is it???
    phi = template_params['coa_phase'] # Is it???

    # First figure out time series that is needed.
    # Demand that 22 mode that is present and use that
    fp = h5py.file(os.path.join(directory)+'Amph22.h5', 'r')
    knots = fp['knots'][:]
    time_start_M = knots[0]
    time_start_s = time_start * lal.MTSUN_SI * total_mass
    time_end_M = knots[-1]
    time_end_s = time_end_M * lal.MTSUN_SI * total_mass

    # Restrict start time if needed
    flow_start_time = seobnrrom_length_in_time(**template_params)
    # t=0 means merger so invert
    flow_start_time = -flow_start_time
    if flow_start_time > time_start_s:
        time_start_s = flow_start_time
        time_start_M = time_start_s / (lal.MTSUN_SI * total_mass)
    else:
        # FIXME: Is this the right behaviour
        err_msg = "WAVEFORM IS NOT LONG ENOUGH. %e %e" \
                                               %(flow_start_time, time_start_s)
        raise ValueError(err_msg)

    # Generate time array
    time_series = numpy.arange(time_start_s, time_end_s, delta_t)
    hp = numpy.zeros(len(time_series), dtype=float)
    hc = numpy.zeros(len(time_series), dtype=float)

    # Generate the waveform
    for l in (2,3,4,5,6,7,8):
        for m in range(-l,l+1):
            amp_file_name = os.path.join(directory, "Amph%d%d.h5" %(l,m))
            phase_file_name = os.path.join(directory, "Phaseh%d%d.h5" %(l,m))
            if not os.path.isfile(amp_file_name):
                continue
            curr_amp = get_data_from_h5_file(amp_file_name, time_series)
            curr_phase = get_data_from_h5_file(phase_file_name, time_series)
            curr_h_real = curr_amp * numpy.cos(curr_phase)
            curr_h_imag = curr_amp * numpy.sin(curr_phase)
            curr_ylm = lal.SpinWeightedSphericalHarmonic(theta, phi, -2, l, m)
            hp += curr_h_real * curr_ylm.real - curr_h_imag * curr_ylm.imag
            # FIXME: No idea whether these should be minus or plus, guessing
            #        minus for now. Only affects some polarization phase
            hc += - curr_h_real * curr_ylm.imag - curr_h_imag * curr_ylm.real 
    return hp, hc
