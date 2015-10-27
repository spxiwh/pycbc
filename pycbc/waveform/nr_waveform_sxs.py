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

from . import UseNRinDA
import nr_waveform


def get_hplus_hcross_from_sxs(hdf5_file_name, template_params, delta_t,\
                                  verbose=False):
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
    total_mass = get_param('mass1') + get_param('mass2')
    theta      = get_param('inclination')
    phi = 0.#template_params['coa_phase']
    distance   = get_param('distance')
    end_time   = get_param('end_time') # FIXME
    
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
    estimated_length = nr_waveform.seobnrrom_length_in_time(**template_params)
    estimated_length_pow2 = 2**ceil(log2( estimated_length * 1.1 ))
    #
    nrwav = UseNRinDA.nr_wave(filename=hdf5_file_name, modeLmax=2, \
                    sample_rate=1./delta_t, time_length=estimated_length_pow2, \
                    totalmass=total_mass, inclination=theta, phi=phi, \
                    distance=distance*1e6,\
                    ex_order=3, \
                    verbose=True)
    
    time_start_s = -nrwav.get_peak_amplitude()[0] * nrwav.dt
    if verbose: print >>sys.stdout, " time_start_s = %f" % time_start_s
    #
    hp = nrwav.rescaled_hp
    hc = nrwav.rescaled_hc
    #
    hpExtraIdx = where(hp.data == 0)[0]
    hcExtraIdx = where(hc.data == 0)[0]
    idx = hpExtraIdx[where(hpExtraIdx == hcExtraIdx)[0][0]]
    #
    #for idx in range(len(hp)):
    #  if hp[idx] == 0 and hc[idx] == 0: break
    #
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

