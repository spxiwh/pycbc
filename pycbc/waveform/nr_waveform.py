# Copyright (C) 2015  Patricia Schmidt, Ian W. Harry
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


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

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
            try:
                return template_params[value]
            except:
                if value in ['spin1x', 'spin1y','spin1z', 'spin2x', 'spin2y', 'spin2']:
                    # Spins are not provided, default to 0
                    return 0.0
            
    mass1 = get_param('mass1')
    mass2 = get_param('mass2')
    total_mass = mass1 + mass2
    
    spin1z = get_param('spin1z')
    spin2z = get_param('spin2z')
    
    spin1x = get_param('spin1x')
    spin2x = get_param('spin2x')
    
    spin1y = get_param('spin1y')
    spin2y = get_param('spin2y')
    
    flower = get_param('f_lower')
    theta = get_param('inclination') # Is it???
    
    # NOTE: This is actually the reference orbital phase of the NR data.
    #       However, the xml table does not know the column phiRef!
    phi = get_param('coa_phase')
    
    end_time = get_param('end_time')
    distance = get_param('distance')
    
    # Open NR file:
    fp = h5py.File(hd5_file_name, 'r')
    
    # Reference frequency:
    #FIXME: Does this mess up xml tables?
    if 'f_ref' in template_params:
        template_params['f_ref'] = fp.attrs['f_lower_at_1MSUN'] / total_mass
    else: template_params['f_ref'] = fp.attrs['f_lower_at_1MSUN'] / total_mass
    f_ref = get_param('f_ref')
    print "The reference frequency has been changed to %1.5f" %f_ref

    # Sanity checking: make sure intrinsic template parameters are consistent
    # with the NR metadata.    
    # FIXME: Add more checks!
    
    # Add check that mass ratio is consistent
    eta = fp.attrs['eta']
    if abs(((mass1 * mass2) / (mass1 + mass2)**2) - eta) > 10**(-3):
        err_msg = "MASSES ARE INCONSISTENT WITH THE MASS RATIO OF THE NR SIMULATION."
        raise ValueError(err_msg)
        
    # Add check that spins are consistent
    if (abs(spin1x - fp.attrs['spin1x']) > 10**(-3) or \
        abs(spin1y - fp.attrs['spin1y']) > 10**(-3) or \
        abs(spin1z - fp.attrs['spin1z']) > 10**(-3) ):
        err_msg = "COMPONENTS OF SPIN1 ARE INCONSISTENT WITH THE NR SIMULATION."
        raise ValueError(err_msg)
    
    if (abs(spin2x - fp.attrs['spin2x']) > 10**(-3) or \
        abs(spin2y - fp.attrs['spin2y']) > 10**(-3) or \
        abs(spin2z - fp.attrs['spin2z']) > 10**(-3) ):
        err_msg = "COMPONENTS OF SPIN2 ARE INCONSISTENT WITH THE NR SIMULATION."
        raise ValueError(err_msg)
    
    # add check that the reference phase is consistent
    #if abs( phi - fp.attrs['coa_phase']) > 10**(-3):
    #    err_msg = "THE COALESCENCE PHASE PARAMETER IS INCORRECT. USE METADATA VALUE."
    #    raise ValueError(err_msg)
        
    # First figure out time series that is needed.
    # Demand that 22 mode that is present and use that
    Mflower = fp.attrs['f_lower_at_1MSUN']
    knots = fp['amp_l2_m2']['knots'][:]
    time_start_M = knots[0]
    time_start_s = time_start_M * lal.MTSUN_SI * total_mass
    print "hybrid start time in M:", time_start_M
    print "hybrdid start time in s:", time_start_s
    time_end_M = knots[-1]
    time_end_s = time_end_M * lal.MTSUN_SI * total_mass

    # Restrict start time if needed
    print template_params
    try:
        est_start_time = seobnrrom_length_in_time(**template_params)
    except:
        est_start_time = -time_start_s
    # t=0 means merger so invert
    est_start_time = -est_start_time
    if est_start_time > time_start_s:
        time_start_s = est_start_time
        time_start_M = time_start_s / (lal.MTSUN_SI * total_mass)
    elif flower < Mflower / total_mass:
        # If waveform is close to full NR length, check against the NR data
        # and either use *all* data, or fail if too short.
        err_msg = "WAVEFORM IS NOT LONG ENOUGH TO REACH f_low. %e %e" \
                                                %(flower, Mflower / total_mass)
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
            print curr_ylm
            hp += curr_h_real * curr_ylm.real + curr_h_imag * curr_ylm.imag
            # FIXME: No idea whether these should be minus or plus, guessing
            #        minus for now. Only affects some polarization phase
            hc += + curr_h_real * curr_ylm.imag - curr_h_imag * curr_ylm.real 

    # Scale by distance
    # FIXME: The original NR scaling is 1M. The steps below scale the distance
    #        appropriately given a total mass M. The distance is now in Mpc. 
    #        Is this the correct unit?
    #       .
    massMpc = total_mass * lal.MRSUN_SI / ( lal.PC_SI * 1.0e6)
    hp *= (massMpc/distance)
    hc *= (massMpc/distance)

    # Time start s is negative and is time from peak to start
    print end_time+time_start_s
    hp = TimeSeries(hp, delta_t=delta_t,
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))
    hc = TimeSeries(hc, delta_t=delta_t,
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))

    fp.close()

    print "Done, returning hp,hc"

    return hp, hc

def get_hplus_hcross_from_get_td_waveform(**p):
    """
    Interface between get_td_waveform and get_hplus_hcross_from_directory above
    """
    delta_t = float(p['delta_t'])
    p['end_time'] = 0.
    
    # Assign correct reference frequency for consistency:
    fp = h5py.File(p['numrel_data'], 'r')
    Mflower = fp.attrs['f_lower_at_1MSUN']
    fp.close()
    mass1 = p['mass1']
    mass2 = p['mass2']
    total_mass = mass1 + mass2
    p['f_ref'] = Mflower / (total_mass)
    print "The reference frequency has been set to %1.5f" %p['f_ref']
    
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
    # FIXME: issues with 10% error margin
    time_length = 1.1 * time_length
    return time_length
