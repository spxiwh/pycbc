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
from . import nr_waveform_sxs

def get_data_from_h5_file(filepointer, time_series, key_name):
    """
    This one is a bit of a Ronseal.
    """
    deg = filepointer[key_name]['deg'][()]
    knots = filepointer[key_name]['knots'][:]
    data = filepointer[key_name]['data'][:]
    # Check time_series is valid
    if knots[0] < 0:
        assert(knots[0]*1.00001 <= time_series[0])
    else:
        assert(knots[0] <= time_series[0]*1.00001)
    if knots[-1] < 0:
        assert(knots[-1] >= time_series[-1]*1.00001)
    else:
        assert(knots[-1]*1.00001 >= time_series[-1])
    spline = UnivariateSpline(knots, data, k=deg, s=0)
    out = spline(time_series, 0)
    return out

def get_rotation_angles_from_h5_file(filepointer, inclination, phi_ref):
    """
    Computes the angles necessary to rotate from the intrinsic NR source frame
    into the LAL frame. See DCC-T1600045 for details.
    Yes, it would be better to make this class-based, however as this has to
    be coded in C it is easier to keep a C-style to make later porting easier.
    """
    cos = numpy.cos
    sin = numpy.sin
    # Following section IV of DCC-T1600045
    # Step 1: Define Phi = phiref/2 ... I'm ignoring this, I think it is wrong
    orb_phase = phi_ref

    # Step 2: Compute Zref
    # 2.1: Compute LN_hat from file. LN_hat = direction of orbital ang. mom.
    ln_hat_x = filepointer.attrs['LNhatx']
    ln_hat_y = filepointer.attrs['LNhaty']
    ln_hat_z = filepointer.attrs['LNhatz']
    ln_hat = numpy.array([ln_hat_x, ln_hat_y, ln_hat_z])
    ln_hat = ln_hat / sum(ln_hat * ln_hat)**0.5

    # 2.2: Compute n_hat from file. n_hat = direction from object 2 to object 1
    n_hat_x = filepointer.attrs['nhatx']
    n_hat_y = filepointer.attrs['nhaty']
    n_hat_z = filepointer.attrs['nhatz']
    n_hat = numpy.array([n_hat_x, n_hat_y, n_hat_z])
    n_hat = n_hat / sum(n_hat*n_hat)**0.5

    # 2.3: Compute Z in the lal wave frame
    corb_phase = cos(orb_phase)
    sorb_phase = sin(orb_phase)
    sinclination = sin(inclination)
    cinclination = cos(inclination)
    ln_cross_n = numpy.cross(ln_hat, n_hat)
    z_wave = sinclination * (sorb_phase * n_hat + corb_phase * ln_cross_n)
    z_wave += cinclination * ln_hat

    # Step 3.1: Extract theta and psi from Z in the lal wave frame
    # NOTE: Theta can only run between 0 and pi, so no problem with arccos here
    theta = numpy.arccos(z_wave[2])
    # Degenerate if Z_wave[2] == 1. In this case just choose psi randomly,
    # the choice will be cancelled out by alpha correction (I hope!)
    if abs(z_wave[2] - 1 ) < 0.000001:
        psi = 0.5
    else:
        # psi can run between 0 and 2pi, but only one solution works for x and y

        # Possible numerical issues if z_wave[0] = sin(theta)
        if abs(z_wave[0] / sin(theta)) > 1:
            if abs(z_wave[0] / sin(theta)) < 1.00001:
                if (z_wave[0] * sin(theta)) < 0.:
                    psi = numpy.pi
                else:
                    psi = 0.
            else:
                err_msg = "Something's bad in Ian's math. Tell him he's an idiot!"
                raise ValueError(err_msg)
        else:
            psi = numpy.arccos(z_wave[0] / sin(theta))
        y_val = sin(psi) * sin(theta)
        # If z_wave[1] is negative, flip psi so that sin(psi) goes negative
        # while preserving cos(psi)
        if z_wave[1] < 0.:
            psi = 2 * numpy.pi - psi
            y_val = sin(psi) * sin(theta)
        if abs(y_val - z_wave[1]) > 0.0001:
            err_msg = "Something's wrong in Ian's math. Tell him he's an idiot!"
            raise ValueError(err_msg)

    # 3.2: Compute the vectors theta_hat and psi_hat
    stheta = sin(theta)
    ctheta = cos(theta)
    spsi = sin(psi)
    cpsi = cos(psi)
    theta_hat = numpy.array([cpsi * ctheta, spsi * ctheta, - stheta])
    psi_hat = numpy.array([-spsi, cpsi, 0])

    # Step 4: Compute sin(alpha) and cos(alpha)
    n_dot_theta = numpy.dot(n_hat, theta_hat)
    ln_cross_n_dot_theta = numpy.dot(ln_cross_n, theta_hat)
    n_dot_psi = numpy.dot(n_hat, psi_hat)
    ln_cross_n_dot_psi = numpy.dot(ln_cross_n, psi_hat)

    calpha = corb_phase * n_dot_theta - sorb_phase * ln_cross_n_dot_theta
    salpha = corb_phase * n_dot_psi - sorb_phase * ln_cross_n_dot_psi

    # Step X: Also useful to keep the source frame vectors as defined in
    #         equation 16 of Harald's document.
    x_source_hat = corb_phase * n_hat - sorb_phase * ln_cross_n
    y_source_hat = sorb_phase * n_hat + corb_phase * ln_cross_n
    z_source_hat = ln_hat
    source_vecs = [x_source_hat, y_source_hat, z_source_hat]

    return theta, psi, calpha, salpha, source_vecs

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
    inclination = get_param('inclination')
    phi_ref = get_param('coa_phase')

    end_time = get_param('end_time')
    distance = get_param('distance')

    # Open NR file:
    fp = h5py.File(hd5_file_name, 'r')

    # Reference frequency:
    if 'f_ref' in template_params:
        # FIXME: If f_ref is given, this should check it is correct!
        template_params['f_ref'] = fp.attrs['f_lower_at_1MSUN'] / total_mass
    else:
        template_params['f_ref'] = fp.attrs['f_lower_at_1MSUN'] / total_mass
    f_ref = get_param('f_ref')

    # Identify rotation parameters. In theory this can be a function of f_ref,
    # but not yet.
    theta, psi, calpha, salpha, source_vecs = \
                  get_rotation_angles_from_h5_file(fp, inclination, phi_ref)

    # Sanity checking: make sure intrinsic template parameters are consistent
    # with the NR metadata.
    # FIXME: Add more checks!

    # Add check that mass ratio is consistent
    eta = fp.attrs['eta']
    if abs(((mass1 * mass2) / (mass1 + mass2)**2) - eta) > 10**(-3):
        err_msg = "MASSES ARE INCONSISTENT WITH THE MASS RATIO OF THE NR SIMULATION."
        raise ValueError(err_msg)

    # Add check that spins are consistent
    #if (abs(spin1x - fp.attrs['spin1x']) > 10**(-3) or \
    #    abs(spin1y - fp.attrs['spin1y']) > 10**(-3) or \
    #    abs(spin1z - fp.attrs['spin1z']) > 10**(-3) ):
    #    err_msg = "COMPONENTS OF SPIN1 ARE INCONSISTENT WITH THE NR SIMULATION."
    #    raise ValueError(err_msg)

    #if (abs(spin2x - fp.attrs['spin2x']) > 10**(-3) or \
    #    abs(spin2y - fp.attrs['spin2y']) > 10**(-3) or \
    #    abs(spin2z - fp.attrs['spin2z']) > 10**(-3) ):
    #    err_msg = "COMPONENTS OF SPIN2 ARE INCONSISTENT WITH THE NR SIMULATION."
    #    raise ValueError(err_msg)

    # First figure out time series that is needed.
    # Demand that 22 mode that is present and use that
    Mflower = fp.attrs['f_lower_at_1MSUN']
    knots = fp['amp_l2_m2']['knots'][:]
    time_start_M = knots[0]
    time_start_s = time_start_M * lal.MTSUN_SI * total_mass
    time_end_M = knots[-1]
    time_end_s = time_end_M * lal.MTSUN_SI * total_mass

    # Restrict start time if needed
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
            curr_amp = get_data_from_h5_file(fp, time_series_M, amp_key)
            curr_phase = get_data_from_h5_file(fp, time_series_M, phase_key)
            curr_h_real = curr_amp * numpy.cos(curr_phase)
            curr_h_imag = curr_amp * numpy.sin(curr_phase)
            curr_ylm = lal.SpinWeightedSphericalHarmonic(theta, psi, -2, l, m)
            # Here is what 0709.0093 defines h_+ and h_x as. This defines the
            # NR wave frame
            curr_hp = curr_h_real * curr_ylm.real - curr_h_imag * curr_ylm.imag
            curr_hc = -curr_h_real*curr_ylm.imag - curr_h_imag * curr_ylm.real

            # Correct for the "alpha" angle as given in T1600045 to translate
            # from the NR wave frame to LAL wave-frame
            hp_corr = (calpha*calpha - salpha*salpha) * curr_hp
            hp_corr += 2 * calpha * salpha * curr_hc
            hc_corr = - 2 * calpha * salpha * curr_hp
            hc_corr += (calpha*calpha - salpha*salpha) * curr_hc
            hp += hp_corr
            hc += hc_corr

    # Scale by distance
    # The original NR scaling is 1M. The steps below scale the distance
    # appropriately given a total mass M. The distance is now in Mpc.
    massMpc = total_mass * lal.MRSUN_SI / ( lal.PC_SI * 1.0e6)
    hp *= (massMpc/distance)
    hc *= (massMpc/distance)

    # Time start s is negative and is time from peak to start
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

    # Re-direct to sxs-format strain reading code
    # For now, if all groups in the hdf file are directories consider that as
    # sufficient evidence that this is a strain file
    if p['approximant'] == 'NR_hdf5_pycbc_sxs':
        hp, hc = nr_waveform_sxs.get_hplus_hcross_from_sxs(p['numrel_data'], p, delta_t)
        return hp, hc
    elif p['approximant'] == 'NR_hdf5_pycbc':
        fp = h5py.File(p['numrel_data'], 'r')

        # Assign correct reference frequency for consistency:
        Mflower = fp.attrs['f_lower_at_1MSUN']
        fp.close()
        mass1 = p['mass1']
        mass2 = p['mass2']
        total_mass = mass1 + mass2
        p['f_ref'] = Mflower / (total_mass)

        hp, hc = get_hplus_hcross_from_directory(p['numrel_data'], p, delta_t)
        return hp, hc
    else:
        err_msg = "Approximant %s not recognized." %(p['approximant'])
        raise ValueError(err_msg)

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
    time_length = 1.1 * time_length
    return time_length
