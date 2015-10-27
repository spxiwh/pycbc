#!/usr/bin/env python
#! /usr/bin/env python
# Copyright (C) 2014 Prayush Kumar
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
#matplotlib.use('Agg')
import os
import sys

from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.misc import derivative
from scipy.optimize import bisect, brentq
from scipy.integrate import simps, cumtrapz

import commands as cmd
import string
from numpy import *
import numpy as np
from numpy.random import random
import h5py

from optparse import OptionParser
from math import pow

#import utils
#phase_from_polarizations = utils.phase_from_polarizations
#frequency_from_polarizations = utils.frequency_from_polarizations
#amplitude_from_polarizations = utils.amplitude_from_polarizations

from utils import phase_from_polarizations, frequency_from_polarizations, amplitude_from_polarizations
from pycbc.types import FrequencySeries, TimeSeries, zeros, real_same_precision_as, complex_same_precision_as, Array
import lal

verbose=True
QM_MTSUN_SI=4.925492321898864e-06


def nextpow2( x ): return int(2**ceil(log2( x )))
  
def planck_window( N=None, eps=None, one_sided=True, winstart=0 ):
  #{{{
  if N is None or eps is None: 
    raise IOError("Please provide the window length and smoothness")
  #
  N = N - winstart
  win = ones(N)
  N1 = int(eps * (N - 1.)) + 1
  den_t1_Zp = 1. + 2. * win / (N - 1.)
  Zp = 2. * eps * (1./den_t1_Zp + 1./(den_t1_Zp - 2.*eps))
  win[0:N1] = array(1. / (exp(Zp) + 1.))[0:N1]
  ##
  if one_sided is not True:
    N2 = (1. - eps) * (N - 1.) + 1
    den_t1_Zm = 1. - 2. * win / (N - 1.)
    Zm = 2. * eps * (1./den_t1_Zm + 1./(den_t1_Zm - 2.*eps))
    win[N2:] = array(1. / (exp(Zm) + 1.))[N2:]
  ##
  win = append( ones(winstart), win )
  return win
  #}}}

def windowing_tanh(waveform_array, bin_to_center_window, sharpness):
  #{{{
  waveform_array = asarray(waveform_array)
  length_of_waveform = size(waveform_array)
  x = arange(length_of_waveform)
  window_function = (tanh(sharpness * (x - bin_to_center_window)) + 1.)/2.
  temp = window_function*waveform_array
  return temp
  #}}} 


def convert_TimeSeries_to_lalREAL8TimeSeries( h, name=None ):
  tmp = lal.CreateREAL8Sequence( len(h) )
  tmp.data = np.array(h.data)
  hnew = lal.REAL8TimeSeries()
  hnew.data = tmp
  hnew.deltaT = h.delta_t
  hnew.epoch = h._epoch
  if name is not None: hnew.name = name
  return hnew

def convert_lalREAL8TimeSeries_to_TimeSeries( h ):
  return TimeSeries(h.data.data, delta_t=h.deltaT, \
                    copy=True, epoch=h.epoch, dtype=h.data.data.dtype)

def zero_pad_beginning( h, steps=1 ):
  h.data = np.roll( h.data, steps )
  return h


######################################################################
######################################################################

#     Make a class for basic manipulations of SpEC NR waveforms      #

######################################################################
######################################################################
class nr_wave():
  #{{{
  def __init__(self, filename=None, filetype='HDF5', wavetype='Auto', \
        ex_order=3, \
        modeLmin=2, modeLmax=8, skipM0=True, \
        sample_rate=8192, time_length=32, rawdelta_t=-1, \
        totalmass=None, inclination=0, phi=0, distance=1.e6, \
        verbose=True):
    """ 
#### Assumptions:
### 1. The nr waveform file is uniformly sampled
### 2. wavetypes passed should be : 
###     CCE , Extrapolated , FiniteRadius , NoGroup, Auto
###     Auto : from filename figure out the file type
### 3. filetypes passed should be : ASCII , HDF5 , DataSet

###### About conventions:
### 1. Modes themselves are not amplitude-scaled. Only their time-axis is
###    rescaled.
### 2. ...

    """
    #
    ##################################################################
    #   0. Ensure inputs are correct
    ##################################################################
    if 'dataset' not in filetype:
      if filename is not None and not os.path.exists(filename):
        raise IOError("Please provide data file!")
      if verbose:
        print >>sys.stderr, "\n Reading From Filename=%s" % filename
    
    # Datafile details
    self.verbose = verbose
    self.filename = filename
    self.filetype = filetype
    self.modeLmax = modeLmax
    self.modeLmin = modeLmin
    self.skipM0   = skipM0
    
    # Extraction parameters
    self.ex_order = ex_order
    
    # Data analysis parameters
    self.sample_rate = sample_rate
    self.time_length = time_length 
    self.dt = 1./self.sample_rate
    self.df = 1./self.time_length
    self.n = self.sample_rate * self.time_length
    if self.verbose: print >>sys.stderr, "self.n = ", self.n
    
    # Binary parameters
    self.totalmass = None
    self.inclination = inclination
    self.phi = phi
    self.distance = distance
    if self.verbose:
      print >>sys.stderr, " >> Input mass, inc, phi, dist = ", totalmass,\
                          self.inclination, self.phi, self.distance
    
    sys.stderr.flush()
    ##################################################################
    #   1. Figure out what the data-storage type (wavetype) is
    ##################################################################
    self.wavetype = None
    if str(wavetype) in ['CCE', 'Extrapolated', 'FiniteRadius', 'NoGroup']:
       self.wavetype = wavetype
    elif str(wavetype) == 'Auto':
      # Decide from filename the wavetype
      fname = self.filename.split('/')[-1]
      if 'rhOverM_Asymptotic_GeometricUnits' in fname:
        self.wavetype = 'Extrapolated'
      elif 'Cce' in fname: self.wavetype = 'CCE'
      elif 'FiniteRadii' in fname: self.wavetype = 'FiniteRadius'
      elif 'HDF' in self.filetype: 
          ftmp = h5py.File(self.filename, 'r')
          fgrps = [str(grptmp) for grptmp in fgrpstmp]
          if 'Y_l2_m2.dat' in fgrps: self.wavetype = 'NoGroup'
    #
    if self.wavetype == None: raise IOError("Could not figure out wavetype")
    if self.verbose: print >>sys.stderr, self.wavetype

    sys.stderr.flush()    
    ##################################################################
    #   2. Read the data from the file. Read all modes.
    ##################################################################
    #
    if 'HDF' in self.filetype:
      if self.verbose: 
        print >>sys.stderr, " > Reading NR data in HDF5 from %s" % self.filename
      
      # Read data file
      self.fin = h5py.File(self.filename,'r')
      if self.wavetype != 'NoGroup':
        grp = self.get_groupname()
        if self.verbose:
          print >>sys.stderr, ("From %s out of " % grp), self.fin.keys()
        wavedata = self.fin[grp]
      else: wavedata = self.fin
      
      # Read modes
      self.rawtsamples, self.rawmodes_real, self.rawmodes_imag = {}, {}, {}
      for modeL in np.arange( 2, self.modeLmax+1 ):
        self.rawtsamples[modeL] = {}
        self.rawmodes_real[modeL], self.rawmodes_imag[modeL] = {}, {}
        for modeM in np.arange( modeL, -1*modeL-1, -1 ):
          if self.skipM0 and modeM==0: continue
          mdata = wavedata['Y_l%d_m%d.dat' % (modeL, modeM)].value
          if self.verbose: print >> sys.stderr, "Reading %d,%d mode" % (modeL, modeM)
          #
          ts = mdata[:,0]
          hp_int = InterpolatedUnivariateSpline(ts, mdata[:,1])
          hc_int = InterpolatedUnivariateSpline(ts, mdata[:,2])
          
          # Hard-coded to re-sample at initial dt
          if rawdelta_t <= 0: self.rawdelta_t = ts[1] - ts[0]
          else: self.rawdelta_t = rawdelta_t
          
          #
          self.rawtsamples[modeL][modeM] = TimeSeries(\
                            np.arange(ts.min(), ts.max(), self.rawdelta_t),\
                            delta_t=self.rawdelta_t, epoch=0)
          self.rawmodes_real[modeL][modeM] = TimeSeries(\
                            array([hp_int(t) for t in self.rawtsamples[2][2]]),\
                            delta_t=self.rawdelta_t, epoch=0)
          self.rawmodes_imag[modeL][modeM] = TimeSeries(\
                            array([hc_int(t) for t in self.rawtsamples[2][2]]),\
                            delta_t=self.rawdelta_t, epoch=0)
    #
    elif 'dataset' in self.filetype:
      raise IOError("datasets not supported yet!")
      data = self.filename
      ts = data[:,0] - data[0,0]
      hp_int = InterpolatedUnivariateSpline(ts, data[:,1])
      hc_int = InterpolatedUnivariateSpline(ts, data[:,2])
      # Hard-coded to re-sample at dt = 1M
      if rawdelta_t <= 0: self.rawdelta_t = 1.
      else: self.rawdelta_t = rawdelta_t
      self.rawtsamples = TimeSeries(arange(0, ts.max()),\
                                    delta_t=self.rawdelta_t, epoch=0)
      self.rawhp = TimeSeries(array([hp_int(t) for t in self.rawtsamples]),\
                                    delta_t=self.rawdelta_t, epoch=0)
      self.rawhc = TimeSeries(array([hc_int(t) for t in self.rawtsamples]),\
                                    delta_t=self.rawdelta_t, epoch=0)      
      print >>sys.stderr, "times go from %f to %f" % (min(ts), max(ts))
      print >>sys.stderr, "rawhp Min = %e, Max = %e" % (min(self.rawhp), max(self.rawhp))
      #
    elif 'ASCII' in self.filetype:
      raise IOError("ASCII datafile not accepted yet!")
      if self.verbose: print >>sys.stderr, "Reading NR data in ASCII from %s" % \
                                        self.filename
      
      # Read modes
      self.rawtsamples, self.rawmodes_real, self.rawmodes_imag = {}, {}, {}
      for modeL in np.arange( 2, self.modeLmax+1 ):
        self.rawtsamples[modeL] = {}
        self.rawmodes_real[modeL], self.rawmodes_imag[modeL] = {}, {}
        for modeM in np.arange( -1*modeL, modeL+1 ):
          if self.skipM0 and modeM==0: continue
          mdata = np.loadtxt(self.filename % (modeL, modeM))
          if self.verbose: print >> sys.stderr, np.shape(mdata)
          #
          ts = mdata[:,0]
          hp_int = InterpolatedUnivariateSpline(ts, mdata[:,1])
          hc_int = InterpolatedUnivariateSpline(ts, mdata[:,2])
          
          # Hard-coded to re-sample at initial dt
          if rawdelta_t <= 0: self.rawdelta_t = ts[1] - ts[0]
          else: self.rawdelta_t = rawdelta_t
          
          #
          self.rawtsamples[modeL][modeM] = TimeSeries(\
                                  np.arange(ts.min(), ts.max(), self.rawdelta_t),\
                                  delta_t=self.rawdelta_t, epoch=0)
          self.rawmodes_real[modeL][modeM] = TimeSeries(\
                                  array([hp_int(t) for t in self.rawtsamples]),\
                                  delta_t=self.rawdelta_t, epoch=0)
          self.rawmodes_imag[modeL][modeM] = TimeSeries(\
                                  array([hc_int(t) for t in self.rawtsamples]),\
                                  delta_t=self.rawdelta_t, epoch=0)                                  
    #
    self.rescaled_hp = None
    self.rescaled_hc = None
    if totalmass:
      self.rescale_to_totalmass(totalmass)
      self.totalmass = totalmass
    
    try: self.fin.close()
    except: pass
  #
  def get_groupname(self):
    #{{{
    f = self.fin
    if self.wavetype == 'CCE':
      grp='CceR%04d.dir' % max([int(k.split('.dir')[0][-4:]) for k in f.keys()])
      return grp
    #
    elif self.wavetype == 'Extrapolated':
      for k in f.keys():
        try: n = int(k[-1])
        except:
          try: n = int(k.split('.dir')[0][-1])
          except:
            if self.verbose:
              print >>sys.stderr, " .. Extrapolated groupname is %" % k
            raise IOError("Could not find the group for extrapolated waveforms")
        if self.ex_order == n: return k
    #
    elif self.wavetype == 'FiniteRadius':
      grp='R%04d.dir' % max([int(k.split('.dir')[0][-4:]) for k in f.keys()])
      return grp
    #
    raise KeyError("Groupname not found")
    #}}}
  # ##################################################################
  # Basic waveform manipulation
  # ##################################################################
  def rescale_mode(self, M=None, distance=None, modeL=2, modeM=2):
    """ This function rescales the given mode to input mass value. No distance
        scaling is done. This function is meant for usage in 
        amplitude-scaling-invariant calculations.
        Note that this function does NOT reset internal total-mass value,
        since it operates on a single mode, and reseting mass/distance etc
        would make things inconsistent
    """
    if (self.totalmass == M or M is None) and self.totalmass != None:
      return [self.rescaledmodes_real[modeL][modeM], \
              self.rescaledmodes_imag[modeL][modeM]]
    
    MinSecs = M * lal.MTSUN_SI
    scaleFac = 1 
    
    if self.verbose: print >>sys.stderr," Rescaling mode %d, %d" % (modeL,modeM)
    rawmode_time = self.rawtsamples[modeL][modeM]
    rawmode_real = self.rawmodes_real[modeL][modeM]
    rawmode_imag = self.rawmodes_imag[modeL][modeM]
    
    end_t = rawmode_time.data[-1] * MinSecs
    start_t = rawmode_time.data[0] * MinSecs
    end_t_n = int((end_t-start_t)/self.dt)
        
    rescaled_hpI = InterpolatedUnivariateSpline(\
                    rawmode_time.data*MinSecs - start_t, rawmode_real.data, k=3)
    rescaled_hcI = InterpolatedUnivariateSpline(\
                    rawmode_time.data*MinSecs - start_t, rawmode_imag.data, k=3)
                                
    tmp_rescaled_hp = rescaled_hpI( np.arange(end_t_n) * self.dt )
    tmp_rescaled_hc = rescaled_hcI( np.arange(end_t_n) * self.dt )
    
    tmp_rescaled_hp = np.concatenate((tmp_rescaled_hp, np.zeros(self.n-end_t_n) ))
    tmp_rescaled_hc = np.concatenate((tmp_rescaled_hc, np.zeros(self.n-end_t_n) ))
        
    if self.verbose: print >>sys.stderr, self.n, end_t_n
    
    return [TimeSeries(tmp_rescaled_hp * scaleFac, delta_t=self.dt, epoch=0),\
            TimeSeries(tmp_rescaled_hc * scaleFac, delta_t=self.dt, epoch=0)]
  #
  def rescale_wave(self, M=None, inclination=None, phi=None, distance=None):
    """ Rescale modes and polarizations to given mass, angles, distance.
        Note that this function re-sets the stored values of binary parameters
        and so all future calculations will assume new values unless otherwise
        specified.
    """
    # if ALL input parameters are the same as internal ones, return stored wave
    # If ALL input parameters are None, return stored wave
    if (self.totalmass == M and self.inclination == inclination and \
        self.phi == phi and self.distance == distance) or \
       (M is None and inclination is None and phi is None and distance is None):
      return [self.rescaled_hp, self.rescaled_hc]
    
    #
    # If MASS has changed, rescale modes
    #
    if self.totalmass != M and M is not None:
      # Rescale the time-axis for all modes
      self.rescaledmodes_real, self.rescaledmodes_imag = {}, {}
      for modeL in np.arange( 2, self.modeLmax+1 ):
        self.rescaledmodes_real[modeL], self.rescaledmodes_imag[modeL] = {}, {}
        for modeM in np.arange( -1*modeL, modeL+1 ):
          if self.skipM0 and modeM==0: continue
          self.rescaledmodes_real[modeL][modeM], \
          self.rescaledmodes_imag[modeL][modeM] = \
                    self.rescale_mode(M, modeL=modeL, modeM=modeM)
      self.totalmass = M
    elif self.totalmass == None and M == None:
      raise IOError("Please provide a total-mass value to rescale")
    
    #
    # Now rescale with distance and ANGLES
    #
    if inclination is not None: self.inclination = inclination
    if phi is not None: self.phi = phi
    if distance is not None: self.distance = distance
    
    # Mass / distance scaling pre-factor
    scalefac = self.totalmass * lal.MRSUN_SI / self.distance / lal.PC_SI
    
    hp, hc = np.zeros(self.n, dtype=float), np.zeros(self.n, dtype=float)
    
    # Combine all modes
    for modeL in np.arange( 2, self.modeLmax+1 ):
      for modeM in np.arange( -1*modeL, modeL+1 ):
        if self.skipM0 and modeM==0: continue
        # h+ - \ii hx = \Sum Ylm * hlm
        curr_ylm = lal.SpinWeightedSphericalHarmonic(\
                        self.inclination, self.phi, -2, modeL, modeM)        
        hp += self.rescaledmodes_real[modeL][modeM].data * curr_ylm.real - \
              self.rescaledmodes_imag[modeL][modeM].data * curr_ylm.imag
        hc += - self.rescaledmodes_real[modeL][modeM].data * curr_ylm.imag - \
                self.rescaledmodes_imag[modeL][modeM].data * curr_ylm.real 
        
    # Scale amplitude by mass and distance factors
    self.rescaled_hp = TimeSeries(scalefac * hp, delta_t=self.dt, epoch=0)
    self.rescaled_hc = TimeSeries(scalefac * hc, delta_t=self.dt, epoch=0)
        
    return [self.rescaled_hp, self.rescaled_hc]
  #
  def rescale_to_totalmass(self, M):
    """ Rescales the waveform to a different total-mass than currently. The 
    values for different angles are set to internal values provided earlier, e.g.
    during object initialization.
    """
    if not hasattr(self, 'inclination') or self.inclination is None:
      raise RuntimeError("Cannot rescale total-mass without setting inclination")
    elif not hasattr(self, 'phi') or self.phi is None:
      raise RuntimeError("Cannot rescale total-mass without setting phi")
    elif not hasattr(self, 'distance') or self.distance is None:
      raise RuntimeError("Cannot rescale total-mass without setting distance")
    
    return self.rescale_wave(M, inclination=self.inclination, phi=self.phi,\
                              distance=self.distance)
  #
  def rescale_to_distance(self, distance):
    """ Rescales the waveform to a different distance than currently. The 
    values for different angles, masses are set to internal values provided 
    earlier, e.g. during object initialization.
    """
    if not hasattr(self, 'inclination') or self.inclination is None:
      raise RuntimeError("Cannot rescale distance without setting inclination")
    elif not hasattr(self, 'phi') or self.phi is None:
      raise RuntimeError("Cannot rescale distance without setting phi")
    elif not hasattr(self, 'totalmass') or self.totalmass is None:
      raise RuntimeError("Cannot rescale distance without setting total-mass")
    
    return self.rescale_wave(self.totalmass, inclination=self.inclination, \
                              phi=self.phi, distance=distance)
  #
  def rotate(self, inclination=0, phi=0):
    """ Rotates waveforms to different inclination and initial-phase angles, 
    with the total-mass and distance set to internal values, provided earlier,
    e.g. during object initialization.
    """
    if not hasattr(self, 'totalmass') or self.totalmass is None:
      raise RuntimeError("Cannot rotate without setting total mass")
    elif not hasattr(self, 'distance') or self.distance is None:
      raise RuntimeError("Cannot rescale total-mass without setting distance")
    
    return self.rescale_wave(self.totalmass, inclination=inclination, phi=phi,\
                            distance=self.distance)
  #
  def get_polarizations(self, M=None, inclination=None, phi=None, distance=None):
    if M is None: M = self.totalmass
    if inclination is None: inclination = self.inclination
    if phi is None: phi = self.phi
    if distance is None: distance = self.distance
    return self.rescale_wave(M, \
                      inclination=inclination, phi=phi, distance=distance)
  #
  def resample(self, dt):
    if not hasattr(self, 'totalmass') or self.totalmass is None:
      raise RuntimeError("Cannot resample without setting total mass")
    elif not hasattr(self, 'inclination') or self.inclination is None:
      raise RuntimeError("Cannot resample total-mass without setting inclination")
    elif not hasattr(self, 'phi') or self.phi is None:
      raise RuntimeError("Cannot resample total-mass without setting phi")
    elif not hasattr(self, 'distance') or self.distance is None:
      raise RuntimeError("Cannot resample total-mass without setting distance")

    if dt == self.dt: return [self.rescaled_hp, self.rescaled_hc]
    else:
      self.dt = dt
      return self.rescale_wave(self.totalmass, inclination=self.inclination,\
                                phi=self.phi, distance=self.distance)
  #
  ####################################################################
  ####################################################################
  ##### Functions to operate on individual modes
  ####################################################################
  ####################################################################
  #
  # Read mode amplitude as a function of t (in s or M)
  #
  def get_mode_amplitude(self, totalmass=None, modeL=2, modeM=2, dimensionless=False):
    """ compute the amplitude of a given mode. If dimensionless amplitude as a
    function of dimensionless time is not needed, make sure totalmass is set 
    either in this function, or in the object earlier.
    """
    if dimensionless:
      hre = self.rawmodes_real[modeL][modeM]
      him = self.rawmodes_imag[modeL][modeM]
    else:
      # If a physical mass has been provided, returned rescaled amplitude
      if totalmass is not None:
        hre, him = self.rescale_mode(totalmass, modeL=modeL, modeM=modeM)
      elif self.totalmass != None:
        hre = self.rescaledmodes_real[modeL][modeM]
        him = self.rescaledmodes_imag[modeL][modeM]
      elif self.totalmass is None:
        raise IOError("Please provide total-mass to rescale modes to")
    return amplitude_from_polarizations( hre, him )
  #
  # Returns the phase (in radians) as a function of t (in s or M)
  #
  def get_mode_phase(self, totalmass=None, modeL=2, modeM=2, dimensionless=False):
    if dimensionless:
      hre = self.rawmodes_real[modeL][modeM]
      him = self.rawmodes_imag[modeL][modeM]
    else:
      # If a physical mass has been provided, returned rescaled amplitude
      if totalmass is not None:
        hre, him = self.rescale_mode(totalmass, modeL=modeL, modeM=modeM)
      elif self.totalmass != None:
        hre = self.rescaledmodes_real[modeL][modeM]
        him = self.rescaledmodes_imag[modeL][modeM]
      elif self.totalmass is None:
        raise IOError("Please provide total-mass to rescale modes to")
    return phase_from_polarizations(hre, him)
  #
  # Returns frequency (in Hz or 1/M) as a function of t (in s or M)
  #
  def get_mode_frequency(self, totalmass=None, modeL=2, modeM=2, dimensionless=False):
    if dimensionless:
      hre = self.rawmodes_real[modeL][modeM]
      him = self.rawmodes_imag[modeL][modeM]
    else:
      # If a physical mass has been provided, return rescaled amplitude
      if totalmass is not None:
        hre, him = self.rescale_mode(totalmass, modeL=modeL, modeM=modeM)
      elif self.totalmass != None:
        hre = self.rescaledmodes_real[modeL][modeM]
        him = self.rescaledmodes_imag[modeL][modeM]
      elif self.totalmass is None:
        raise IOError("Please provide total-mass to rescale modes to")
    
    return frequency_from_polarizations(hre, him)
  #
  # Returns amplitude, phase and frequency all in one
  #
  def get_mode_amplitude_phase_frequency(self, totalmass=None,\
                                      modeL=2, modeM=2, dimensionless=False):
    amp = self.get_mode_amplitude(totalmass=totalmass, \
                          modeL=modeL, modeM=modeM, dimensionless=dimensionless)
    phs = self.get_mode_phase( totalmass=totalmass, \
                          modeL=modeL, modeM=modeM, dimensionless=dimensionless)
    frq = self.get_mode_frequency(totalmass=totalmass, \
                          modeL=modeL, modeM=modeM, dimensionless=dimensionless)
    return [amp, phs, frq]
  #
  ####################################################################
  ####################################################################
  ##### Functions to operate on polarizations
  ####################################################################
  ####################################################################
  #
  # Amplitude of polarization
  # Defaults of "None" for input parameters mean "use those already provided"
  #
  def get_polarization_amplitude(self, totalmass=None, inclination=None, \
                                  phi=None, distance=None):
    if totalmass is None: totalmass = self.totalmass
    if inclination is None: inclination = self.inclination
    if phi is None: phi = self.phi
    if distance is None: distance = self.distance
    
    if totalmass is None: raise IOError("Please provide totalmass")
    if inclination is None: raise IOError("Please provide inclination")
    if phi is None: raise IOError("Please provide initial phase angle")
    if distance is None: raise IOError("Please provide source distance")
    
    hp, hc = self.rescale_wave( totalmass, inclination=inclination,
                                  phi=phi, distance=distance )
    
    amp = amplitude_from_polarizations(hp, hc)
    return amp
  #
  # Phase of the complex polarization
  #
  def get_polarization_phase(self, totalmass=None, inclination=None, \
                                  phi=None, distance=None):
    if totalmass is None: totalmass = self.totalmass
    if inclination is None: inclination = self.inclination
    if phi is None: phi = self.phi
    if distance is None: distance = self.distance
    
    if totalmass is None: raise IOError("Please provide totalmass")
    if inclination is None: raise IOError("Please provide inclination")
    if phi is None: raise IOError("Please provide initial phase angle")
    if distance is None: raise IOError("Please provide source distance")
    
    hp, hc = self.rescale_wave( totalmass, inclination=inclination,
                                  phi=phi, distance=distance )
    
    phase = phase_from_polarizations(hp, hc)
    return phase
  #
  # Derivative of the polarization phase
  #
  def get_polarization_frequency(self, totalmass=None, inclination=None, \
                                  phi=None, distance=None):
    if totalmass is None: totalmass = self.totalmass
    if inclination is None: inclination = self.inclination
    if phi is None: phi = self.phi
    if distance is None: distance = self.distance
    
    if totalmass is None: raise IOError("Please provide totalmass")
    if inclination is None: raise IOError("Please provide inclination")
    if phi is None: raise IOError("Please provide initial phase angle")
    if distance is None: raise IOError("Please provide source distance")
    
    hp, hc = self.rescale_wave( totalmass, inclination=inclination,
                                  phi=phi, distance=distance )
    
    freq = frequency_from_polarizations(hp, hc)
    return freq
  #
  ####################################################################
  # Functions related to wave-frequency
  ####################################################################
  #
  # Get dimensionless orbital frequency as a function of time (s)
  #
  def get_orbital_omega(self, dimensionless=True):
    if not dimensionless: raise IOError("Only orbital M*Omega available")
    freq = self.get_mode_frequency(dimensionless=dimensionless)
    freq.data = freq.data * np.pi
    return freq
  #
  # Get 2,2-mode GW_frequency in Hz at a given time (M)
  #
  def get_frequency_t(self, t, totalmass=60.):
    """ Get 2,2-mode GW_frequency in Hz at a given time (M) """
    self.rescale_to_totalmass(totalmass)
    mf = self.get_mode_frequency(dimensionless=False)
    #
    index = int(np.round(t * totalmass * lal.MTSUN_SI * self.sample_rate))
    if self.verbose:
      print >>sys.stderr, "> get_orbital_frequency:: index = %d, freq = %f" % \
                          (index, mf.data[index])
    return totalmass, mf.data[index]
  #
  # Get the GW frequency at t = 0
  #
  def get_starting_frequency(self, totalmass=None):
    """ # Get the GW frequency at t = 0. This implementation is naive. """
    return self.get_frequency_t(t, totalmass=totalmass)
  #
  # Get the 2,2-mode GW frequency in Hz at the peak of |h22|
  #
  def get_peak_frequency(self, totalmass=None):
    """ Get the 2,2-mode GW frequency in Hz at the peak of |h22|. """
    if totalmass is None:
      if self.totalmass is not None: totalmass = self.totalmass
      else: raise IOError("Need to set the total-mass first")
    
    amp = self.get_mode_amplitude(totalmass=totalmass, dimensionless=False)
    frq = self.get_mode_frequency(totalmass=totalmass, dimensionless=False)
    #iStart = int((self.rawdelta_t*totalmass*QM_MTSUN_SI/hp0.delta_t)*len(self.rawtsamples)*3./4.)
    iStart = int((self.rawtsamples[2][2][-1] - self.rawtsamples[2][2][0])*3./4. * totalmass * lal.MTSUN_SI * self.sample_rate)
    for idx in range(iStart, len(amp)):
      if amp[idx+1] < amp[idx]: break
    iMax = idx
    return [iMax, frq.data[iMax]]
  def get_lowest_binary_mass(self, t, f_lower):
    """This function gives the total mass corresponding to a given starting time
    in the NR waveform, and a desired physical lower frequency cutoff.
    t = units of Total Mass
    """
    rescaled_mass, orbit_freq1 = self.get_frequency_t(t)
    m_lower = orbit_freq1 * rescaled_mass / f_lower
    return m_lower
  #
  ###################################################################
  # Functions related to wave-amplitude
  ###################################################################
  #
  #
  # Get the 2,2-mode GW amplitude at the peak of |h22|
  #
  def get_peak_amplitude(self, totalmass=None):
    """ Get the 2,2-mode GW amplitude at the peak of |h22|. """
    if totalmass is None:
      if self.totalmass is not None: totalmass = self.totalmass
      else: raise IOError("Need to set the total-mass first")
    
    amp = self.get_mode_amplitude(totalmass=totalmass, dimensionless=False)
    iStart = int((self.rawtsamples[2][2][-1] - self.rawtsamples[2][2][0])*3./4. * totalmass * lal.MTSUN_SI * self.sample_rate)
    int(len(self.rawtsamples[2][2]) * 3./4.)
    for idx in range(iStart, len(amp)):
      if amp[idx+1] < amp[idx]: break
    iMax = idx
    return [iMax, amp.data[iMax]]
  #
  ###################################################################
  ####################################################################
  # Strain conditioning
  ####################################################################
  ###################################################################
  def window_waveform(self, hpsamp=None, hcsamp=None, eps=0.001, winstart=0, wintype="planck-taper"):
    """Window the waveform to reduce frequency oscillations at the beginning"""
    if 'planck' not in wintype: 
      raise IOError("Only Planck-taper window available")
    if hpsamp and hcsamp: hp0, hc0 = [hpsamp, hcsamp]
    elif self.rescaled_hp is not None and self.rescaled_hc is not None:
      totalmass = self.totalmass
      hp0, hc0 = [self.rescaled_hp, self.rescaled_hc]
    else: raise IOError("Please provide either the total mass (and strain)")
    #
    for i in range( len(hp0) ):
      if hp0[i]==0 and hc0[i]==0: break
    N = i #len(hp0)
    window_array = planck_window( N=N, eps=eps, winstart=winstart )
    if len(window_array) < len(hp0):
      window_array = append(window_array, ones(len(hp0) - len(window_array)))
    #    
    hpret = TimeSeries(window_array*hp0.data, \
        dtype=real_same_precision_as(hp0), delta_t=hp0.delta_t, copy=True)
    hcret = TimeSeries(window_array*hc0.data, \
        dtype=real_same_precision_as(hc0), delta_t=hp0.delta_t, copy=True)
    return [hpret, hcret]
  #
  def taper_filter_waveform( self, hpsamp=None, hcsamp=None, \
            ntaper1=100, ntaper2=1100, ntaper3=-1, ntaper4=-1, npad=00, f_filter=10. ):
    """Tapers using a cosine (Tukey) window and high-pass filters"""
    if hpsamp and hcsamp: hp0, hc0 = [hpsamp, hcsamp]
    elif self.rescaled_hp is not None and self.rescaled_hc is not None:
      totalmass = self.totalmass
      hp0, hc0 = [self.rescaled_hp, self.rescaled_hc]
    else: raise IOError("Please provide either the total mass (and strain)")
    # Check windowing extents
    if ntaper1 > ntaper2 or ntaper2 > ntaper3 or ntaper3 > ntaper4:
      raise IOError("Invalid window configuration with [%d,%d,%d,%d]" %\
        (ntaper1,ntaper2,ntaper3,ntaper4))
    #
    hp = TimeSeries( hp0.data, dtype=hp0.dtype, delta_t=hp0.delta_t, epoch=hp0._epoch )
    hc = TimeSeries( hc0.data, dtype=hc0.dtype, delta_t=hc0.delta_t, epoch=hc0._epoch )
    # Get actual waveform length
    for idx in np.arange( len(hp)-1, 0, -1 ):
      if hp[idx]==0 and hc[idx]==0: break
    N = idx #np.where( hp.data == 0 )[0][0]
    # Check npad
    if abs(len(hp) - N) < npad:
      print >>sys.stdout, "Ignoring npad..\n"
      npad = 0
    else:
      # Prepend some zeros to the waveform (assuming there are ''npad'' zeros at the end)
      hp = zero_pad_beginning( hp, steps=npad )
      hc = zero_pad_beginning( hc, steps=npad )
    #
    # Construct the taper window
    win = np.zeros(npad+ntaper1) # padded portion
    win12 = 0.5 + 0.5*np.array([np.cos( np.pi*(float(j-ntaper1)/float(ntaper2-\
                          ntaper1) - 1)) for j in np.arange(ntaper1,ntaper2)])
    win = np.append(win, win12)
    win23 = np.ones(ntaper3-ntaper2)
    win = np.append(win, win23)
    win34 = 0.5 - 0.5*np.array([np.cos( np.pi*(float(j-ntaper3)/float(ntaper4-\
                          ntaper3) - 1)) for j in np.arange(ntaper3,ntaper4)])
    win = np.append(win, win34)
    win4N = np.zeros(len(hp)-ntaper4)
    win = np.append(win, win4N)
    # Taper the waveform
    hp.data *= win
    hc.data *= win
    #
    # High pass filter the waveform
    hplal = convert_TimeSeries_to_lalREAL8TimeSeries( hp )
    hclal = convert_TimeSeries_to_lalREAL8TimeSeries( hc )
    lal.HighPassREAL8TimeSeries( hplal, f_filter, 0.9, 8 )
    lal.HighPassREAL8TimeSeries( hclal, f_filter, 0.9, 8 )
    hpnew = convert_lalREAL8TimeSeries_to_TimeSeries( hplal )
    hcnew = convert_lalREAL8TimeSeries_to_TimeSeries( hclal )
    return hpnew, hcnew
  #
  def blending_function_Tukey( self, hp0, t, sample_rate, time_length, f_filter=14. ):
    t1,t2,t3,t4 = t[0]*lal.MTSUN_SI,t[1]*lal.MTSUN_SI,t[2]*lal.MTSUN_SI,t[3]*lal.MTSUN_SI
    i1,i2,i3,i4 = int(t1*sample_rate),int(t2*sample_rate),int(t3*sample_rate),int(t4*sample_rate)
    # Return if window specs are impossible
    if not (i1 < i2 and i2 < i3 and i3 < i4):
      print t1, t2, t3, t4
      print i1, i2, i3, i4
      raise IOError( "Invalid window configuration" )
      return hp0
    hpnew, _ = self.taper_filter_waveform(hpsamp=hp0, hcsamp=hp0, \
                        ntaper1=i1, ntaper2=i2, ntaper3=i3, ntaper4=i4, \
                        f_filter=f_filter)
    return hpnew
  #
  def blending_function( self, hp0, t, sample_rate, time_length, f_filter=14. ):
    # Blending function of the waveform using McKechan's function (2010 paper) and high pass filter
    # h0 is TimeSeries - rescaled to some mass
    # works for real part only, or you could do real and imaginary part separately if you need both
    # t is a length-4 array of t1, t2, t3, t4 in SOLAR MASSES
    t1,t2,t3,t4 = t[0]*lal.MTSUN_SI,t[1]*lal.MTSUN_SI,t[2]*lal.MTSUN_SI,t[3]*lal.MTSUN_SI
    i1,i2,i3,i4 = int(t1*sample_rate),int(t2*sample_rate),int(t3*sample_rate),int(t4*sample_rate)
    #print t
    #print t1,t2,t3,t4
    #print i1,i2,i3,i4
    time_array = hp0.sample_times.data
    #
    # Return if window specs are impossible
    if not (i1 < i2 and i2 < i3 and i3 < i4):
      print t1, t2, t3, t4
      print i1, i2, i3, i4
      raise IOError( "Invalid window configuration" )
      return hp0
    #
    region1 = np.zeros(i1)          # = 0 for t<t1
    region2 = np.zeros(i2-i1)       # = 1/(exp(blah)+1) for t1<t<t2
    region3 = np.ones(i3-i2)        # = 1 for t2<t<t3
    region4 = np.zeros(i4-i3)       # = 1/(exp(blah)+1) for t3<t<t4
    region5 = np.zeros(len(hp0)-i4) # = 0 for t>t4
    #
    np.seterr(divide='raise',over='raise',under='raise',invalid='raise')
    for i in range(len(region2)):
      try:
        region2[i] = 1./(np.exp( ((t2-t1)/(time_array[i+i1]-t1)) + ((t2-t1)/(time_array[i+i1]-t2)) ) + 1)
      except:
        if time_array[i+i1]>0.9*t1 and time_array[i+i1]<1.1*t1:
          region2[i] = 0
        if time_array[i+i1]>0.9*t2 and time_array[i+i1]<1.1*t2:
          region2[i] = 1.
    for i in range(len(region4)):
      try:
        region4[i] = 1./(np.exp( ((t3-t4)/(time_array[i+i3]-t3)) + ((t3-t4)/(time_array[i+i3]-t4)) ) + 1)
      except:
        if time_array[i+i3]>0.9*t3 and time_array[i+i3]<1.1*t3:
          region4[i] = 1.
        if time_array[i+i3]>0.9*t4 and time_array[i+i3]<1.1*t4:
          region4[i] = 0
    func = np.concatenate((region1,region2,region3,region4,region5)) # combine regions into one array
    hp_blended = np.zeros(len(hp0.data))
    for i in range(len(func)):
      try:
        hp_blended[i] = func[i]*hp0.data[i] # creates array of blended data
    # hc_blended = func*hc0.data
      except:
        hp_blended[i] = 0
    hp = TimeSeries(hp_blended,dtype=hp0.dtype,delta_t=hp0.delta_t,epoch=hp0._epoch) # turn back into TimeSeries
    # hc = TimeSeries(hc_blended,dtype=hc0.dtype,delta_t=hc0.delta_t,epoch=hc0._epoch)
    #
    # High pass filter the waveform
    hplal = convert_TimeSeries_to_lalREAL8TimeSeries( hp )
    # hclal = convert_TimeSeries_to_lalREAL8TimeSeries( hc )
    lal.HighPassREAL8TimeSeries( hplal, f_filter, 0.9, 8 )
    # lal.HighPassREAL8TimeSeries( hclal, f_filter, 0.9, 8 )
    hpnew = convert_lalREAL8TimeSeries_to_TimeSeries( hplal )
    # hcnew = convert_lalREAL8TimeSeries_to_TimeSeries( hclal )
    return hpnew
  #}}}


