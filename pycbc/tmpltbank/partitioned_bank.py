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

import copy
import numpy
import logging

from pycbc.tmpltbank import coord_utils

logger = logging.getLogger('pycbc.tmpltbank.partitioned_bank')

class PartitionedTmpltbank(object):
    """
    This class is used to hold a template bank partitioned into numerous bins
    based on position in the Cartesian parameter space where the axes are the
    principal components. It can also be used to hold intermediary
    products used while constructing (e.g.) a stochastic template bank.
    """
    def __init__(self, mass_range_params, metric_params, ref_freq,
                 bin_spacing, bin_range_check=1):
        """
        Set up the partitioned template bank class. The combination of the
        reference frequency, the bin spacing and the metric dictates how the
        parameter space will be partitioned.

        Parameters
        -----------
        mass_range_params : massRangeParameters object
            An initialized massRangeParameters object holding the details of
            the mass and spin ranges being considered.
        metric_params : metricParameters object
            An initialized metricParameters object holding the details of the
            parameter space metric that is being used.
        ref_freq : float
            The reference frequency to use as the upper frequency cutoff of
            the metric when partitioning the bank. In general this would be
            set to the *smallest* upper frequency cutoff that is possible in
            the given parameter space. However, in some cases this can lead
            to only a small number of partitions and the computational cost
            will increase dramatically. NOTE: when using the vary-fupper
            option this upper frequency cutoff is only used to determine which
            points should be matched against each other, it is *not* used in
            the actual metric-based calculation of the distance (which uses the
            frequency cutoffs of the points being considered).
        bin_spacing : float
            The metric distance to space the bins by. NOTE: If you want to
            place the bins to have a width corresponding to a minimal match of
            0.97 you would set this to :math:`(1 - 0.97)^{0.5}`.
            Note the square root,
            matches correspond to the square of parameter space distance.
        bin_range_check : int
            When computing matches consider points in the corresponding bin and
            all bins +/- this value in both chi_1 and chi_2 directions.
            DEFAULT = 1.
        """
        # Flags to be used in other methods of this class. Initialized here for
        # simplicity
        self.spin_warning_given = False

        # These will probably be used a lot, so add to object
        self.mass_range_params = mass_range_params
        self.metric_params = metric_params
        self.ref_freq = ref_freq
        self.bin_spacing = bin_spacing

        # Get parameter space extent
        vals = coord_utils.estimate_mass_range(1000000, mass_range_params,
                                          metric_params, ref_freq, covary=True)
        chi1_max = vals[0].max()
        chi1_min = vals[0].min()
        chi1_diff = chi1_max - chi1_min
        chi2_max = vals[1].max()
        chi2_min = vals[1].min()
        chi2_diff = chi2_max - chi2_min
        # Add a little bit extra as we may not have reached the edges.
        # FIXME: Maybe better to use the numerical code to find maxima here?
        chi1_min = chi1_min - 0.1*chi1_diff
        chi1_max = chi1_max + 0.1*chi1_diff
        chi2_min = chi2_min - 0.1*chi2_diff
        chi2_max = chi2_max + 0.1*chi2_diff

        massbank = {}
        bank = {}
        # Also add a little bit here
        for i in range(-2, int((chi1_max - chi1_min) // bin_spacing + 2)):
            bank[i] = {}
            massbank[i] = {}
            for j in range(-2, int((chi2_max - chi2_min) // bin_spacing + 2)):
                bank[i][j] = []
                massbank[i][j] = {}
                massbank[i][j]['mass1s'] = numpy.array([])

        self.massbank = massbank
        self.bank = bank
        # Record minimum and maximum bins
        self.min_chi1_bin = -2
        self.min_chi2_bin = -2
        self.max_chi1_bin = int((chi1_max - chi1_min) // bin_spacing + 1)
        self.max_chi2_bin = int((chi2_max - chi2_min) // bin_spacing + 1)
        self.chi1_min = chi1_min
        self.chi1_max = chi1_max
        self.chi2_min = chi2_min
        self.chi2_max = chi2_max

        # How many adjacent bins should we check?
        self.bin_range_check = 1
        self.bin_loop_order = coord_utils.outspiral_loop(self.bin_range_check)

    def get_point_from_bins_and_idx(self, chi1_bin, chi2_bin, idx):
        """Find masses and spins given bin numbers and index.

        Given the chi1 bin, chi2 bin and an index, return the masses and spins
        of the point at that index. Will fail if no point exists there.

        Parameters
        -----------
        chi1_bin : int
            The bin number for chi1.
        chi2_bin : int
            The bin number for chi2.
        idx : int
            The index within the chi1, chi2 bin.

        Returns
        --------
        mass1 : float
            Mass of heavier body.
        mass2 : float
            Mass of lighter body.
        spin1z : float
            Spin of heavier body.
        spin2z : float
            Spin of lighter body.
        """
        mass1 = self.massbank[chi1_bin][chi2_bin]['mass1s'][idx]
        mass2 = self.massbank[chi1_bin][chi2_bin]['mass2s'][idx]
        spin1z = self.massbank[chi1_bin][chi2_bin]['spin1s'][idx]
        spin2z = self.massbank[chi1_bin][chi2_bin]['spin2s'][idx]
        return mass1, mass2, spin1z, spin2z

    def get_freq_map_and_normalizations(self, frequency_list,
                                        upper_freq_formula):
        """
        If using the --vary-fupper capability we need to store the mapping
        between index and frequencies in the list. We also precalculate the
        normalization factor at every frequency, which is used when estimating
        overlaps to account for abrupt changes in termination frequency.

        Parameters
        -----------
        frequency_list : array of floats
            The frequencies for which the metric has been computed and lie
            within the parameter space being considered.
        upper_freq_formula : string
        """
        self.frequency_map = {}
        self.normalization_map = {}
        self.upper_freq_formula = upper_freq_formula
        # FIXME: Must this be sorted on input
        frequency_list.sort()

        for idx, frequency in enumerate(frequency_list):
            self.frequency_map[frequency] = idx
            self.normalization_map[frequency] = \
                             (self.metric_params.moments['I7'][frequency])**0.5

    def find_point_bin(self, chi_coords):
        """
        Given a set of coordinates in the chi parameter space, identify the
        indices of the chi1 and chi2 bins that the point occurs in. Returns
        these indices.

        Parameters
        -----------
        chi_coords : numpy.array
            The position of the point in the chi coordinates.

        Returns
        --------
        chi1_bin : int
            Index of the chi_1 bin.
        chi2_bin : int
            Index of the chi_2 bin.
        """
        # Identify bin
        chi1_bin = int((chi_coords[0] - self.chi1_min) // self.bin_spacing)
        chi2_bin = int((chi_coords[1] - self.chi2_min) // self.bin_spacing)
        self.check_bin_existence(chi1_bin, chi2_bin)
        return chi1_bin, chi2_bin


    def check_bin_existence(self, chi1_bin, chi2_bin):
        """
        Given indices for bins in chi1 and chi2 space check that the bin
        exists in the object. If not add it. Also check for the existence of
        all bins within +/- self.bin_range_check and add if not present.

        Parameters
        -----------
        chi1_bin : int
            The index of the chi1_bin to check
        chi2_bin : int
            The index of the chi2_bin to check
        """
        bin_range_check = self.bin_range_check
        # Check if this bin actually exists. If not add it
        if ( (chi1_bin < self.min_chi1_bin+bin_range_check) or
             (chi1_bin > self.max_chi1_bin-bin_range_check) or
             (chi2_bin < self.min_chi2_bin+bin_range_check) or
             (chi2_bin > self.max_chi2_bin-bin_range_check) ):
            for temp_chi1 in range(chi1_bin-bin_range_check,
                                                   chi1_bin+bin_range_check+1):
                if temp_chi1 not in self.massbank:
                    self.massbank[temp_chi1] = {}
                    self.bank[temp_chi1] = {}
                for temp_chi2 in range(chi2_bin-bin_range_check,
                                                   chi2_bin+bin_range_check+1):
                    if temp_chi2 not in self.massbank[temp_chi1]:
                        self.massbank[temp_chi1][temp_chi2] = {}
                        self.massbank[temp_chi1][temp_chi2]['mass1s'] =\
                                                                numpy.array([])
                        self.bank[temp_chi1][temp_chi2] = []

    def calc_point_distance(self, chi_coords):
        """
        Calculate distance between point and the bank. Return the closest
        distance.

        Parameters
        -----------
        chi_coords : numpy.array
            The position of the point in the chi coordinates.

        Returns
        --------
        min_dist : float
            The smallest **SQUARED** metric distance between the test point and
            the bank.
        indexes : The chi1_bin, chi2_bin and position within that bin at which
            the closest matching point lies.
        """
        chi1_bin, chi2_bin = self.find_point_bin(chi_coords)
        min_dist = 1000000000
        indexes = None
        for chi1_bin_offset, chi2_bin_offset in self.bin_loop_order:
            curr_chi1_bin = chi1_bin + chi1_bin_offset
            curr_chi2_bin = chi2_bin + chi2_bin_offset
            for idx, bank_chis in \
                            enumerate(self.bank[curr_chi1_bin][curr_chi2_bin]):
                dist = coord_utils.calc_point_dist(chi_coords, bank_chis)
                if dist < min_dist:
                    min_dist = dist
                    indexes = (curr_chi1_bin, curr_chi2_bin, idx)
        return min_dist, indexes

    def test_point_distance(self, chi_coords, distance_threshold):
        """
        Test if the distance between the supplied point and the bank is less
        than the supplied distance theshold.

        Parameters
        -----------
        chi_coords : numpy.array
            The position of the point in the chi coordinates.
        distance_threshold : float
            The **SQUARE ROOT** of the metric distance to test as threshold.
            E.g. if you want to test to a minimal match of 0.97 you would
            use 1 - 0.97 = 0.03 for this value.

        Returns
        --------
        Boolean
            True if point is within the distance threshold. False if not.

        """
        chi1_bin, chi2_bin = self.find_point_bin(chi_coords)
        for chi1_bin_offset, chi2_bin_offset in self.bin_loop_order:
            curr_chi1_bin = chi1_bin + chi1_bin_offset
            curr_chi2_bin = chi2_bin + chi2_bin_offset
            for bank_chis in self.bank[curr_chi1_bin][curr_chi2_bin]:
                dist = coord_utils.calc_point_dist(chi_coords, bank_chis)
                if dist < distance_threshold:
                    return True
        else:
            return False

    def calc_point_distance_vary(self, chi_coords, point_fupper, mus):
        """
        Calculate distance between point and the bank allowing the metric to
        vary based on varying upper frequency cutoff. Slower than
        calc_point_distance, but more reliable when upper frequency cutoff can
        change a lot.

        Parameters
        -----------
        chi_coords : numpy.array
            The position of the point in the chi coordinates.
        point_fupper : float
            The upper frequency cutoff to use for this point. This value must
            be one of the ones already calculated in the metric.
        mus : numpy.array
            A 2D array where idx 0 holds the upper frequency cutoff and idx 1
            holds the coordinates in the [not covaried] mu parameter space for
            each value of the upper frequency cutoff.

        Returns
        --------
        min_dist : float
            The smallest **SQUARED** metric distance between the test point and
            the bank.
        indexes : The chi1_bin, chi2_bin and position within that bin at which
            the closest matching point lies.
        """
        chi1_bin, chi2_bin = self.find_point_bin(chi_coords)
        min_dist = 1000000000
        indexes = None
        for chi1_bin_offset, chi2_bin_offset in self.bin_loop_order:
            curr_chi1_bin = chi1_bin + chi1_bin_offset
            curr_chi2_bin = chi2_bin + chi2_bin_offset
            # No points = Next iteration
            curr_bank = self.massbank[curr_chi1_bin][curr_chi2_bin]
            if not curr_bank['mass1s'].size:
                continue

            # *NOT* the same of .min and .max
            f_upper = numpy.minimum(point_fupper, curr_bank['freqcuts'])
            f_other = numpy.maximum(point_fupper, curr_bank['freqcuts'])
            # NOTE: freq_idxes is a vector!
            freq_idxes = numpy.array([self.frequency_map[f] for f in f_upper])
            # vecs1 gives a 2x2 vector: idx0 = stored index, idx1 = mu index
            vecs1 = mus[freq_idxes, :]
            # vecs2 gives a 2x2 vector: idx0 = stored index, idx1 = mu index
            range_idxes = numpy.arange(len(freq_idxes))
            vecs2 = curr_bank['mus'][range_idxes, freq_idxes, :]

            # Now do the sums
            dists = (vecs1 - vecs2)*(vecs1 - vecs2)
            # This reduces to 1D: idx = stored index
            dists = numpy.sum(dists, axis=1)
            norm_upper = numpy.array([self.normalization_map[f] \
                                      for f in f_upper])
            norm_other = numpy.array([self.normalization_map[f] \
                                      for f in f_other])
            norm_fac = norm_upper / norm_other
            renormed_dists = 1 - (1 - dists)*norm_fac
            curr_min_dist = renormed_dists.min()
            if curr_min_dist < min_dist:
                min_dist = curr_min_dist
                indexes = curr_chi1_bin, curr_chi2_bin, renormed_dists.argmin()

        return min_dist, indexes

    def test_point_distance_vary(self, chi_coords, point_fupper, mus,
                                 distance_threshold):
        """
        Test if distance between point and the bank is greater than distance
        threshold while allowing the metric to
        vary based on varying upper frequency cutoff. Slower than
        test_point_distance, but more reliable when upper frequency cutoff can
        change a lot.

        Parameters
        -----------
        chi_coords : numpy.array
            The position of the point in the chi coordinates.
        point_fupper : float
            The upper frequency cutoff to use for this point. This value must
            be one of the ones already calculated in the metric.
        mus : numpy.array
            A 2D array where idx 0 holds the upper frequency cutoff and idx 1
            holds the coordinates in the [not covaried] mu parameter space for
            each value of the upper frequency cutoff.
        distance_threshold : float
            The **SQUARE ROOT** of the metric distance to test as threshold.
            E.g. if you want to test to a minimal match of 0.97 you would
            use 1 - 0.97 = 0.03 for this value.

        Returns
        --------
        Boolean
            True if point is within the distance threshold. False if not.
        """
        chi1_bin, chi2_bin = self.find_point_bin(chi_coords)
        for chi1_bin_offset, chi2_bin_offset in self.bin_loop_order:
            curr_chi1_bin = chi1_bin + chi1_bin_offset
            curr_chi2_bin = chi2_bin + chi2_bin_offset
            # No points = Next iteration
            curr_bank = self.massbank[curr_chi1_bin][curr_chi2_bin]
            if not curr_bank['mass1s'].size:
                continue

            # *NOT* the same of .min and .max
            f_upper = numpy.minimum(point_fupper, curr_bank['freqcuts'])
            f_other = numpy.maximum(point_fupper, curr_bank['freqcuts'])
            # NOTE: freq_idxes is a vector!
            freq_idxes = numpy.array([self.frequency_map[f] for f in f_upper])
            # vecs1 gives a 2x2 vector: idx0 = stored index, idx1 = mu index
            vecs1 = mus[freq_idxes, :]
            # vecs2 gives a 2x2 vector: idx0 = stored index, idx1 = mu index
            range_idxes = numpy.arange(len(freq_idxes))
            vecs2 = curr_bank['mus'][range_idxes,freq_idxes,:]

            # Now do the sums
            dists = (vecs1 - vecs2)*(vecs1 - vecs2)
            # This reduces to 1D: idx = stored index
            dists = numpy.sum(dists, axis=1)
            # I wonder if this line actually speeds things up?
            if (dists > distance_threshold).all():
                continue
            # This is only needed for close templates, should we prune?
            norm_upper = numpy.array([self.normalization_map[f] \
                                      for f in f_upper])
            norm_other = numpy.array([self.normalization_map[f] \
                                      for f in f_other])
            norm_fac = norm_upper / norm_other
            renormed_dists = 1 - (1 - dists)*norm_fac
            if (renormed_dists < distance_threshold).any():
                return True
        else:
            return False

    def add_point_by_chi_coords(self, chi_coords, mass1, mass2, spin1z, spin2z,
                          point_fupper=None, mus=None):
        """
        Add a point to the partitioned template bank. The point_fupper and mus
        kwargs must be provided for all templates if the vary fupper capability
        is desired. This requires that the chi_coords, as well as mus and
        point_fupper if needed, to be precalculated. If you just have the
        masses and don't want to worry about translations see
        add_point_by_masses, which will do translations and then call this.

        Parameters
        -----------
        chi_coords : numpy.array
            The position of the point in the chi coordinates.
        mass1 : float
            The heavier mass of the point to add.
        mass2 : float
            The lighter mass of the point to add.
        spin1z: float
            The [aligned] spin on the heavier body.
        spin2z: float
            The [aligned] spin on the lighter body.
            The upper frequency cutoff to use for this point. This value must
            be one of the ones already calculated in the metric.
        mus : numpy.array
            A 2D array where idx 0 holds the upper frequency cutoff and idx 1
            holds the coordinates in the [not covaried] mu parameter space for
            each value of the upper frequency cutoff.

        chi_coords may instead be a (n_chi, N) array, with mass1..spin2z
        equal-length arrays, to add N points together in bulk.
        """
        chi_coords = numpy.asarray(chi_coords)
        if chi_coords.ndim > 1:
            self._add_points_by_chi_coords(chi_coords, mass1, mass2, spin1z,
                                           spin2z)
            return

        chi1_bin, chi2_bin = self.find_point_bin(chi_coords)
        self.bank[chi1_bin][chi2_bin].append(copy.deepcopy(chi_coords))
        curr_bank = self.massbank[chi1_bin][chi2_bin]

        if curr_bank['mass1s'].size:
            curr_bank['mass1s'] = numpy.append(curr_bank['mass1s'],
                                               numpy.array([mass1]))
            curr_bank['mass2s'] = numpy.append(curr_bank['mass2s'],
                                               numpy.array([mass2]))
            curr_bank['spin1s'] = numpy.append(curr_bank['spin1s'],
                                               numpy.array([spin1z]))
            curr_bank['spin2s'] = numpy.append(curr_bank['spin2s'],
                                               numpy.array([spin2z]))
            if point_fupper is not None:
                curr_bank['freqcuts'] = numpy.append(curr_bank['freqcuts'],
                                                 numpy.array([point_fupper]))
            # Mus needs to append onto axis 0. See below for contents of
            # the mus variable
            if mus is not None:
                curr_bank['mus'] = numpy.append(curr_bank['mus'],
                                            numpy.array([mus[:,:]]), axis=0)
        else:
            curr_bank['mass1s'] = numpy.array([mass1])
            curr_bank['mass2s'] = numpy.array([mass2])
            curr_bank['spin1s'] = numpy.array([spin1z])
            curr_bank['spin2s'] = numpy.array([spin2z])
            if point_fupper is not None:
                curr_bank['freqcuts'] = numpy.array([point_fupper])
            # curr_bank['mus'] is a 3D array
            # NOTE: mu relates to the non-covaried Cartesian coordinate system
            # Axis 0: Template index
            # Axis 1: Frequency cutoff index
            # Axis 2: Mu coordinate index
            if mus is not None:
                curr_bank['mus'] = numpy.array([mus[:,:]])

    def _add_points_by_chi_coords(self, chi_coords, mass1s, mass2s, spin1zs,
                                  spin2zs):
        """Add many points at once given their chi coordinates.

        chi_coords has shape (n_chi, N). Points are grouped by bin with a
        single sort, so each bin's arrays are written once as a contiguous
        slice rather than grown one element at a time with numpy.append (which
        is quadratic in bin occupancy). Any templates already in a bin are
        appended to.
        """
        n_points = chi_coords.shape[1]
        if not n_points:
            return
        chi1_bins = ((chi_coords[0] - self.chi1_min)
                     // self.bin_spacing).astype(int)
        chi2_bins = ((chi_coords[1] - self.chi2_min)
                     // self.bin_spacing).astype(int)
        # Sort so points sharing a bin are contiguous.
        order = numpy.lexsort((chi2_bins, chi1_bins))
        c1_sorted = chi1_bins[order]
        c2_sorted = chi2_bins[order]
        chi_sorted = chi_coords.T[order]
        m1_sorted = numpy.asarray(mass1s)[order]
        m2_sorted = numpy.asarray(mass2s)[order]
        s1_sorted = numpy.asarray(spin1zs)[order]
        s2_sorted = numpy.asarray(spin2zs)[order]
        # Start index of each contiguous bin group.
        change = numpy.ones(n_points, dtype=bool)
        change[1:] = ((c1_sorted[1:] != c1_sorted[:-1]) |
                      (c2_sorted[1:] != c2_sorted[:-1]))
        starts = numpy.flatnonzero(change)
        ends = numpy.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = n_points
        # Make sure every occupied bin (and its neighbours) exists.
        for s in starts:
            self.check_bin_existence(int(c1_sorted[s]), int(c2_sorted[s]))
        # Store each bin's templates, appending to anything already present.
        for s, e in zip(starts, ends):
            c1 = int(c1_sorted[s])
            c2 = int(c2_sorted[s])
            new_chi = chi_sorted[s:e]
            curr = self.bank[c1][c2]
            if isinstance(curr, list):
                if curr:
                    self.bank[c1][c2] = numpy.vstack(
                        [numpy.asarray(curr), new_chi])
                else:
                    self.bank[c1][c2] = new_chi
            else:
                self.bank[c1][c2] = numpy.vstack([curr, new_chi])
            curr_bank = self.massbank[c1][c2]
            if curr_bank['mass1s'].size:
                curr_bank['mass1s'] = numpy.concatenate(
                    [curr_bank['mass1s'], m1_sorted[s:e]])
                curr_bank['mass2s'] = numpy.concatenate(
                    [curr_bank['mass2s'], m2_sorted[s:e]])
                curr_bank['spin1s'] = numpy.concatenate(
                    [curr_bank['spin1s'], s1_sorted[s:e]])
                curr_bank['spin2s'] = numpy.concatenate(
                    [curr_bank['spin2s'], s2_sorted[s:e]])
            else:
                curr_bank['mass1s'] = m1_sorted[s:e]
                curr_bank['mass2s'] = m2_sorted[s:e]
                curr_bank['spin1s'] = s1_sorted[s:e]
                curr_bank['spin2s'] = s2_sorted[s:e]

    def add_point_by_masses(self, mass1, mass2, spin1z, spin2z,
                            vary_fupper=False):
        """
        Add a point to the template bank. This differs from add point to bank
        as it assumes that the chi coordinates and the products needed to use
        vary_fupper have not already been calculated. This function calculates
        these products and then calls add_point_by_chi_coords.
        This function also
        carries out a number of sanity checks (eg. is the point within the
        ranges given by mass_range_params) that add_point_by_chi_coords does
        not do for speed concerns.

        Parameters
        -----------
        mass1 : float
            Mass of the heavier body
        mass2 : float
            Mass of the lighter body
        spin1z : float
            Spin of the heavier body
        spin2z : float
            Spin of the lighter body

        The four mass/spin arguments may equally be equal-length arrays, in
        which case all points are added together (much faster than one call
        per template).
        """
        # Array inputs: add every point in one vectorized pass.
        if numpy.ndim(mass1):
            mass1 = numpy.asarray(mass1)
            mass2 = numpy.asarray(mass2)
            spin1z = numpy.asarray(spin1z)
            spin2z = numpy.asarray(spin2z)
            if not len(mass1):
                return
            if vary_fupper:
                # vary-fupper needs per-point mu/fupper products; add singly.
                for idx in range(len(mass1)):
                    self.add_point_by_masses(mass1[idx], mass2[idx],
                                             spin1z[idx], spin2z[idx],
                                             vary_fupper=True)
                return
            if numpy.any(mass2 > mass1) and not self.spin_warning_given:
                logger.warning("Adding templates where mass2 > mass1. The "
                               "convention is mass1 > mass2. This message "
                               "will not be repeated.")
                self.spin_warning_given = True
            if numpy.any(self.mass_range_params.is_outside_range(
                    mass1, mass2, spin1z, spin2z)):
                raise ValueError("One or more templates are not consistent "
                                 "with the provided command-line restrictions "
                                 "on masses and spins.")
            # Vectorized metric transform, chunked to bound peak memory.
            n_points = len(mass1)
            chunk = 2000000
            chi_coords = None
            for start in range(0, n_points, chunk):
                end = min(start + chunk, n_points)
                cc = numpy.array(coord_utils.get_cov_params(
                    mass1[start:end], mass2[start:end], spin1z[start:end],
                    spin2z[start:end], self.metric_params, self.ref_freq))
                if chi_coords is None:
                    chi_coords = numpy.empty((cc.shape[0], n_points))
                chi_coords[:, start:end] = cc
            self.add_point_by_chi_coords(chi_coords, mass1, mass2, spin1z,
                                         spin2z)
            return

        # Test that masses are the expected way around (ie. mass1 > mass2)
        if mass2 > mass1:
            if not self.spin_warning_given:
                warn_msg = "Am adding a template where mass2 > mass1. The "
                warn_msg += "convention is that mass1 > mass2. Swapping mass1 "
                warn_msg += "and mass2 and adding point to bank. This message "
                warn_msg += "will not be repeated."
                logger.warning(warn_msg)
                self.spin_warning_given = True

        # These that masses obey the restrictions of mass_range_params
        if self.mass_range_params.is_outside_range(mass1, mass2, spin1z,
                                                                       spin2z):
            err_msg = "Point with masses given by "
            err_msg += "%f %f %f %f " %(mass1, mass2, spin1z, spin2z)
            err_msg += "(mass1, mass2, spin1z, spin2z) is not consistent "
            err_msg += "with the provided command-line restrictions on masses "
            err_msg += "and spins."
            raise ValueError(err_msg)

        # Get chi coordinates
        chi_coords = coord_utils.get_cov_params(mass1, mass2, spin1z, spin2z,
                                                self.metric_params,
                                                self.ref_freq)

        # Get mus and best fupper for this point, if needed
        if vary_fupper:
            mass_dict = {}
            mass_dict['m1'] = numpy.array([mass1])
            mass_dict['m2'] = numpy.array([mass2])
            mass_dict['s1z'] = numpy.array([spin1z])
            mass_dict['s2z'] = numpy.array([spin2z])
            freqs = numpy.array(list(self.frequency_map.keys()), dtype=float)
            freq_cutoff = coord_utils.return_nearest_cutoff(\
                                     self.upper_freq_formula, mass_dict, freqs)
            freq_cutoff = freq_cutoff[0]
            lambdas = coord_utils.get_chirp_params\
                (mass1, mass2, spin1z, spin2z, self.metric_params.f0,
                 self.metric_params.pnOrder)
            mus = []
            for freq in self.frequency_map:
                mus.append(coord_utils.get_mu_params(lambdas,
                                                    self.metric_params, freq) )
            mus = numpy.array(mus)
        else:
            freq_cutoff=None
            mus=None

        self.add_point_by_chi_coords(chi_coords, mass1, mass2, spin1z, spin2z,
                               point_fupper=freq_cutoff, mus=mus)


    def add_tmpltbank_from_xml_table(self, sngl_table, vary_fupper=False):
        """
        This function will take a sngl_inspiral_table of templates and add them
        into the partitioned template bank object.

        Parameters
        -----------
        sngl_table : sngl_inspiral_table
            List of sngl_inspiral templates.
        vary_fupper : False
            If given also include the additional information needed to compute
            distances with a varying upper frequency cutoff.
        """
        mass1s = numpy.array([sngl.mass1 for sngl in sngl_table])
        mass2s = numpy.array([sngl.mass2 for sngl in sngl_table])
        spin1zs = numpy.array([sngl.spin1z for sngl in sngl_table])
        spin2zs = numpy.array([sngl.spin2z for sngl in sngl_table])
        self.add_point_by_masses(mass1s, mass2s, spin1zs, spin2zs,
                                 vary_fupper=vary_fupper)

    def add_tmpltbank_from_hdf_file(self, hdf_fp, vary_fupper=False):
        """
        This function will take a pointer to an open HDF File object containing
        a list of templates and add them into the partitioned template bank
        object.

        Parameters
        -----------
        hdf_fp : h5py.File object
            The template bank in HDF5 format.
        vary_fupper : False
            If given also include the additional information needed to compute
            distances with a varying upper frequency cutoff.
        """
        mass1s = hdf_fp['mass1'][:]
        mass2s = hdf_fp['mass2'][:]
        spin1zs = hdf_fp['spin1z'][:]
        spin2zs = hdf_fp['spin2z'][:]
        self.add_point_by_masses(mass1s, mass2s, spin1zs, spin2zs,
                                 vary_fupper=vary_fupper)

    def output_all_points(self):
        """Return all points in the bank.

        Return all points in the bank as lists of m1, m2, spin1z, spin2z.

        Returns
        -------
        mass1 : list
            List of mass1 values.
        mass2 : list
            List of mass2 values.
        spin1z : list
            List of spin1z values.
        spin2z : list
            List of spin2z values.
        """
        mass1 = []
        mass2 = []
        spin1z = []
        spin2z = []
        for i in self.massbank.keys():
            for j in self.massbank[i].keys():
                for k in range(len(self.massbank[i][j]['mass1s'])):
                    curr_bank = self.massbank[i][j]
                    mass1.append(curr_bank['mass1s'][k])
                    mass2.append(curr_bank['mass2s'][k])
                    spin1z.append(curr_bank['spin1s'][k])
                    spin2z.append(curr_bank['spin2s'][k])

        return mass1, mass2, spin1z, spin2z
