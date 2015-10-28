#
# Copyright (C) 2015  Prayush Kumar
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

import os, sys
from numpy import where, ceil, log2

import lal
from pycbc.types import TimeSeries
from pycbc.pnutils import mtotal_eta_to_mass1_mass2

#import nr_waveform
from . import UseNRinDA

MAX_NR_LENGTH = 100000

def get_hplus_hcross_from_sxs(hdf5_file_name, template_params, delta_t,\
                                  taper=True, verbose=False):
    if verbose:
        print >>sys.stdout, " \n\n\nIn get_hplus_hcross_from_sxs.."
        sys.stdout.flush()
    #
    def get_param(value):
        try:
            if value == 'end_time':
                # FIXME: Imprecise!
                return float(template_params.get_end())
            else:
                return getattr(template_params, value)
        except:
            return template_params[value]
    #
    # Get relevant binary parameters
    #
    mass1      = get_param('mass1')
    mass2      = get_param('mass2')
    total_mass = mass1 + mass2
    theta      = get_param('inclination')
    phi = 0.#template_params['coa_phase']
    distance   = get_param('distance')
    end_time   = get_param('end_time') # FIXME
    f_lower     = get_param('f_lower')
    
    try: taper = get_param('taper')
    except: pass
    try: verbose= get_param('verbose')
    except: pass

    if verbose:
        print >> sys.stdout, "mass, theta, phi , distance, end_time = ", \
                          total_mass, theta, phi, distance, end_time
        try:
          print >>sys.stdout, \
                "end_time = 0 (could be %f)" % template_params['end_time']
        except: pass
    #
    # Figure out how much memory to allocate 
    #
    #estimated_length = nr_waveform.seobnrrom_length_in_time(**template_params)
    estimated_length_pow2 = 2**ceil(log2(MAX_NR_LENGTH*total_mass*lal.MTSUN_SI))
    if verbose:
      print "estimated used = ", estimated_length_pow2
    #
    # Read in the waveform from file & rescale it
    #
    idx = 0
    while True:
      try:
        if verbose: print "\n >>try %d at reading waveform" % idx
        idx += 1
        nrwav = UseNRinDA.nr_wave(filename=hdf5_file_name, modeLmax=2, \
                    sample_rate=1./delta_t, time_length=estimated_length_pow2, \
                    totalmass=total_mass, inclination=theta, phi=phi, \
                    distance=distance*1e6,\
                    ex_order=3, \
                    verbose=verbose)
        break
      except ValueError:
        estimated_length_pow2 *= 2

    if verbose: 
      print >> sys.stdout, "Waveform read from %s" % hdf5_file_name
      sys.stdout.flush()
    #
    # Condition the waveform
    #
    upwin_t_start = 100
    upwin_t_width = 1000
    downwin_amp_frac = 0.01
    downwin_t_width = 50
    t_filter    = upwin_t_start + upwin_t_width
    m_lower = nrwav.get_lowest_binary_mass( t_filter, f_lower)
    
    # Check if re-scaling to input total mass is allowed by the tapering choice
    if m_lower > total_mass:
      raise IOError("Can rescale down to %f Msun at %fHz, asked for %f Msun" % \
                            (m_lower, f_lower, total_mass))

    # Taper the waveforma
    if taper:
      hp, hc = nrwav.taper_filter_waveform(ttaper1=upwin_t_start,\
                    ttaper2=upwin_t_width, ftaper3=downwin_amp_frac,\
                    ttaper4=downwin_t_width,\
                    verbose=verbose)
    else:
      hp, hc = [nrwav.rescaled_hp, nrwav.rescaled_hc]

    time_start_s = -nrwav.get_peak_amplitude()[0] * nrwav.dt
    if verbose: 
      print >>sys.stdout, " time_start_s = %f" % time_start_s
      sys.stdout.flush()
    #
    # Chop off trailing zeros in the waveform
    #
    i_filter = int(ceil( t_filter * total_mass * lal.MTSUN_SI / nrwav.dt ))
    hpExtraIdx = where(hp.data[i_filter:] == 0)[0]
    hcExtraIdx = where(hc.data[i_filter:] == 0)[0]
    idx = i_filter + hpExtraIdx[where(hpExtraIdx == hcExtraIdx)[0][0]]
    #
    if verbose:
      print >>sys.stdout, " Index = %d where waveform ends" % idx
      sys.stdout.flush()

    # FIXME: Correct for length to peak of h22 amplitude, instead of ..
    hp = TimeSeries(hp.data[:idx], delta_t=delta_t,\
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))
    hc = TimeSeries(hc.data[:idx], delta_t=delta_t,\
                    epoch=lal.LIGOTimeGPS(end_time+time_start_s))
    #
    if verbose:
      try:
        print >>sys.stdout, \
                "  Length of rescaled waveform = %f.." % idx*hp.delta_t
        print >>sys.stdout, " hp.epoch = ", hp._epoch
        sys.stdout.flush()
      except: print type(idx), type(hp)
    #
    if verbose: print "REturning hp, hc"
    #
    return hp, hc

