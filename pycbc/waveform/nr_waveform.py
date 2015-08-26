# Add copyright stuff

import os
import h5py
from scipy.interpolate import UnivariateSpline

import numpy

import lal
import lalsimulation

from pycbc.types import TimeSeries

def get_data_from_h5_file(filepointer, time_series, key_name):
    """
    This one is a bit of a Ronseal.
    """
    deg = filepointer[key_name]['deg'][()]
    knots = filepointer[key_name]['knots'][:]
    data = filepointer[key_name]['data'][:]
    # Check time_series is valid
    assert(knots[0] <= time_series[0])
    assert(knots[-1] >= time_series[-1])
    spline = UnivariateSpline(knots, data, k=deg, s=0)
    out = spline(time_series, 0)
    return out

def get_hplus_hcross_from_directory(hd5_file_name, template_params, delta_t):
    """
    Generate hplus, hcross from a NR hd5 file, for a given total mass and
    f_lower.
    """
    def get_param(value):
        try:
            if value == 'end_time':
                # FIXME: Imprecise!
                return float(template_params.get_end())
            else:
                return getattr(template_params, value)
        except:
            return template_params[value]

    mass1 = get_param('mass1')
    mass2 = get_param('mass2')
    total_mass = mass1 + mass2
    spin1z = get_param('spin2z')
    spin2z = get_param('spin2z')
    flower = get_param('f_lower')
    theta = get_param('inclination') # Is it???
    phi = get_param('coa_phase') # Is it???
    end_time = get_param('end_time')
    print end_time
    distance = get_param('distance')

    # Sanity checking
    # FIXME: Add more checks!
    # FIXME: Add check that mass ratio is consistent
    # FIXME: Add check that spins are consistent

    # First figure out time series that is needed.
    # Demand that 22 mode that is present and use that
    fp = h5py.File(hd5_file_name, 'r')
    knots = fp['amp_l2_m2']['knots'][:]
    time_start_M = knots[0]
    time_start_s = time_start_M * lal.MTSUN_SI * total_mass
    print "hybrid start time in M:", time_start_M
    print "hybrdid start time in s:", time_start_s
    time_end_M = knots[-1]
    time_end_s = time_end_M * lal.MTSUN_SI * total_mass

    # Restrict start time if needed
    flow_start_time = lalsimulation.SimInspiralChirpTimeBound(flower,
                          mass1*lal.MSUN_SI, mass2*lal.MSUN_SI, spin1z, spin2z)
    #flow_start_time = seobnrrom_length_in_time(**template_params)
    print -flow_start_time
    print time_start_s
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
    time_series_M = time_series / (lal.MTSUN_SI * total_mass)
    hp = numpy.zeros(len(time_series), dtype=float)
    hc = numpy.zeros(len(time_series), dtype=float)

    # Generate the waveform
    # FIXME: should parse list of (l,m)-pairs
    #   IWH: Code currently checks for existence of all modes, and includes a
    #        mode if it is present is this not the right behaviour? If not,
    #        what is?
    for l in (2,3,4,5,6,7,8):
        for m in range(-l,l+1):
            amp_key = 'amp_l%d_m%d' %(l,m)
            phase_key = 'phase_l%d_m%d' %(l,m)
            if amp_key not in fp.keys() or phase_key not in fp.keys():
                continue
            # FIXME: Debugging
            print "Using %d,%d mode" %(l,m)

            curr_amp = get_data_from_h5_file(fp, time_series_M, amp_key)
            curr_phase = get_data_from_h5_file(fp, time_series_M, phase_key)
            curr_h_real = curr_amp * numpy.cos(curr_phase)
            curr_h_imag = curr_amp * numpy.sin(curr_phase)
            curr_ylm = lal.SpinWeightedSphericalHarmonic(theta, phi, -2, l, m)
            hp += curr_h_real * curr_ylm.real - curr_h_imag * curr_ylm.imag
            # FIXME: No idea whether these should be minus or plus, guessing
            #        minus for now. Only affects some polarization phase
            hc += - curr_h_real * curr_ylm.imag - curr_h_imag * curr_ylm.real 

    # Scale by distance
    # FIXME: Is this right, what is the NR waveform scaling?? Probably need
    #        to conver to physical units anyway! (distance is in Mpc)
    hp /= distance
    hc /= distance

    # Time start s is negative and is time from peak to start
    print end_time+time_start_s
    hp = TimeSeries(hp, delta_t=delta_t,
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))
    hc = TimeSeries(hc, delta_t=delta_t,
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))

    fp.close()

    return hp, hc

def get_hplus_hcross_from_get_td_waveform(**p):
    """
    Interface between get_td_waveform and get_hplus_hcross_from_directory above
    """
    delta_t = float(p['delta_t'])
    p['end_time'] = 0.
    hp, hc = get_hplus_hcross_from_directory(p['numrel_data'], p, delta_t)
    return hp, hc

def seobnrrom_length_in_time(**kwds):
    """
    This is a stub for holding the calculation for getting length of the ROM
    waveforms.
    """
    mass1 = kwds['mass1']
    mass2 = kwds['mass2']
    spin1z = kwds['spin1z']
    spin2z = kwds['spin2z']
    fmin = kwds['f_lower']
    chi = lalsimulation.SimIMRPhenomBComputeChi(mass1, mass2, spin1z, spin2z)
    time_length = lalsimulation.SimIMRSEOBNRv2ChirpTimeSingleSpin(
                               mass1*lal.MSUN_SI, mass2*lal.MSUN_SI, chi, fmin)
    # FIXME: This is still approximate so add a 10% error margin
    time_length = time_length * 1.1
    return time_length
