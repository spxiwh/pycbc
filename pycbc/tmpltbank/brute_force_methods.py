"""Numerical routines for the aligned-spin geometric bank placement used by
pycbc_geom_aligned_2dstack. They map a point in the xi coordinate system back
to physical masses and spins, and measure the extent ("depth") of the higher
xi directions at a placed point so that templates can be stacked there.

Two families are provided:

* The deterministic default: a damped Gauss-Newton inversion
  (:func:`get_physical_covaried_masses_newton`) and a predictor-corrector
  continuation depth search (:func:`stack_xi_direction_continuation`). These
  exploit the near-affine structure of the xi coordinates, so they are fast
  and reproducible.
* The older stochastic routines (:func:`get_physical_covaried_masses` and
  :func:`stack_xi_direction_brute`), which throw random points at the space.
  They are much slower and non-deterministic, and are kept for reference and
  reachable via ``pycbc_geom_aligned_2dstack --use-legacy-method``.
"""

import logging
import numpy

from pycbc.tmpltbank.coord_utils import get_cov_params

logger = logging.getLogger('pycbc.tmpltbank.brute_force_methods')


def get_physical_covaried_masses(xis, bestMasses, bestXis, req_match,
                                 massRangeParams, metricParams, fUpper,
                                 giveUpThresh = 5000):
    """
    This function takes the position of a point in the xi parameter space and
    iteratively finds a close point in the physical coordinate space (masses
    and spins).
 
    Parameters
    -----------
    xis : list or array
        Desired position of the point in the xi space. If only N values are
        provided and the xi space's dimension is larger then it is assumed that
        *any* value in the remaining xi coordinates is acceptable.
    bestMasses : list
        Contains [totalMass, eta, spin1z, spin2z]. Is a physical position
        mapped to xi coordinates in bestXis that is close to the desired point.
        This is aimed to give the code a starting point.
    bestXis : list
        Contains the position of bestMasses in the xi coordinate system.
    req_match : float
        Desired maximum mismatch between xis and the obtained point. If a point
        is found with mismatch < req_match immediately stop and return that
        point. A point with this mismatch will not always be found.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    giveUpThresh : int, optional (default = 5000)
        The program will try this many iterations. If no close matching point
        has been found after this it will give up.

    Returns
    --------
    mass1 : float
        The heavier mass of the obtained point.
    mass2 : float
        The smaller mass of the obtained point
    spin1z : float
        The heavier bodies spin of the obtained point.
    spin2z : float
        The smaller bodies spin of the obtained point.
    count : int
        How many iterations it took to find the point. For debugging.
    mismatch : float
        The mismatch between the obtained point and the input xis.
    new_xis : list
        The position of the point in the xi space
    """
    # TUNABLE PARAMETERS GO HERE!
    # This states how far apart to scatter test points in the first proposal
    origScaleFactor = 1

    # Set up
    xi_size = len(xis)
    scaleFactor = origScaleFactor
    bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
    count = 0
    unFixedCount = 0
    currDist = 100000000000000000
    while(1):
        # If we are a long way away we use larger jumps
        if count:
            if currDist > 1 and scaleFactor == origScaleFactor:
                scaleFactor = origScaleFactor*10
        # Get a set of test points with mass -> xi mappings
        totmass, eta, spin1z, spin2z, mass1, mass2, new_xis = \
            get_mass_distribution([bestChirpmass, bestMasses[1], bestMasses[2],
                                   bestMasses[3]],
                                  scaleFactor, massRangeParams, metricParams,
                                  fUpper)
        cDist = (new_xis[0] - xis[0])**2
        for j in range(1,xi_size):
            cDist += (new_xis[j] - xis[j])**2
        if (cDist.min() < req_match):
            idx = cDist.argmin()
            scaleFactor = origScaleFactor
            new_xis_list = [new_xis[ldx][idx] for ldx in range(len(new_xis))]
            return mass1[idx], mass2[idx], spin1z[idx], spin2z[idx], count, \
                   cDist.min(), new_xis_list
        if (cDist.min() < currDist):
            idx = cDist.argmin()
            bestMasses[0] = totmass[idx]
            bestMasses[1] = eta[idx]
            bestMasses[2] = spin1z[idx]
            bestMasses[3] = spin2z[idx]
            bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
            currDist = cDist.min()
            unFixedCount = 0
            scaleFactor = origScaleFactor
        count += 1
        unFixedCount += 1
        if unFixedCount > giveUpThresh:
            # Stop at this point
            diff = (bestMasses[0]*bestMasses[0] * (1-4*bestMasses[1]))**0.5
            mass1 = (bestMasses[0] + diff)/2.
            mass2 = (bestMasses[0] - diff)/2.
            new_xis_list = [new_xis[ldx][0] for ldx in range(len(new_xis))]
            return mass1, mass2, bestMasses[2], bestMasses[3], count, \
                   currDist, new_xis_list
        if not unFixedCount % 100:
            scaleFactor *= 2
        if scaleFactor > 64:
            scaleFactor = 1
    # Shouldn't be here!
    raise RuntimeError

def get_mass_distribution(bestMasses, scaleFactor, massRangeParams,
                          metricParams, fUpper,
                          numJumpPoints=100, chirpMassJumpFac=0.0001,
                          etaJumpFac=0.01, spin1zJumpFac=0.01,
                          spin2zJumpFac=0.01):
    """
    Given a set of masses, this function will create a set of points nearby
    in the mass space and map these to the xi space.

    Parameters
    -----------
    bestMasses : list
        Contains [ChirpMass, eta, spin1z, spin2z]. Points will be placed around
        tjos
    scaleFactor : float
        This parameter describes the radius away from bestMasses that points
        will be placed in.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    numJumpPoints : int, optional (default = 100)
        The number of points that will be generated every iteration
    chirpMassJumpFac : float, optional (default=0.0001)
        The jump points will be chosen with fractional variation in chirpMass
        up to this multiplied by scaleFactor.
    etaJumpFac : float, optional (default=0.01)
        The jump points will be chosen with fractional variation in eta
        up to this multiplied by scaleFactor.
    spin1zJumpFac : float, optional (default=0.01)
        The jump points will be chosen with absolute variation in spin1z up to
        this multiplied by scaleFactor.
    spin2zJumpFac : float, optional (default=0.01)
        The jump points will be chosen with absolute variation in spin2z up to
        this multiplied by scaleFactor.

    Returns 
    --------
    Totmass : numpy.array
        Total mass of the resulting points
    Eta : numpy.array
        Symmetric mass ratio of the resulting points
    Spin1z : numpy.array
        Spin of the heavier body of the resulting points
    Spin2z : numpy.array
        Spin of the smaller body of the resulting points
    Diff : numpy.array
        Mass1 - Mass2 of the resulting points
    Mass1 : numpy.array
        Mass1 (mass of heavier body) of the resulting points
    Mass2 : numpy.array
        Mass2 (mass of smaller body) of the resulting points
    new_xis : list of numpy.array
        Position of points in the xi coordinates
    """
    # FIXME: It would be better if rejected values could be drawn from the 
    # full possible mass/spin distribution. However speed in this function is
    # a major factor and must be considered.
    bestChirpmass = bestMasses[0]
    bestEta = bestMasses[1]
    bestSpin1z = bestMasses[2]
    bestSpin2z = bestMasses[3]

    # Firstly choose a set of values for masses and spins
    chirpmass = bestChirpmass * (1 - (numpy.random.random(numJumpPoints)-0.5) \
                                       * chirpMassJumpFac * scaleFactor )
    etaRange = massRangeParams.maxEta - massRangeParams.minEta
    currJumpFac = etaJumpFac * scaleFactor
    if currJumpFac > etaRange:
        currJumpFac = etaRange
    eta = bestEta * ( 1 - (numpy.random.random(numJumpPoints) - 0.5) \
                           * currJumpFac)

    maxSpinMag = max(massRangeParams.maxNSSpinMag, massRangeParams.maxBHSpinMag)
    minSpinMag = min(massRangeParams.maxNSSpinMag, massRangeParams.maxBHSpinMag)
    # Note that these two are cranged by spinxzFac, *not* spinxzFac/spinxz
    currJumpFac = spin1zJumpFac * scaleFactor
    if currJumpFac > maxSpinMag:
        currJumpFac = maxSpinMag

    # Actually set the new spin trial points
    if massRangeParams.nsbhFlag or (maxSpinMag == minSpinMag):
        curr_spin_1z_jump_fac = currJumpFac
        curr_spin_2z_jump_fac = currJumpFac
        # Check spins aren't going to be unphysical
        if currJumpFac > massRangeParams.maxBHSpinMag:
            curr_spin_1z_jump_fac = massRangeParams.maxBHSpinMag
        if currJumpFac > massRangeParams.maxNSSpinMag:
            curr_spin_2z_jump_fac = massRangeParams.maxNSSpinMag
        spin1z = bestSpin1z + ( (numpy.random.random(numJumpPoints) - 0.5) \
                            * curr_spin_1z_jump_fac)
        spin2z = bestSpin2z + ( (numpy.random.random(numJumpPoints) - 0.5) \
                            * curr_spin_2z_jump_fac)
    else:
        # If maxNSSpinMag is very low (0) and maxBHSpinMag is high we can
        # find it hard to place any points. So mix these when
        # masses are swapping between the NS and BH.
        curr_spin_bh_jump_fac = currJumpFac
        curr_spin_ns_jump_fac = currJumpFac
        # Check spins aren't going to be unphysical
        if currJumpFac > massRangeParams.maxBHSpinMag:
            curr_spin_bh_jump_fac = massRangeParams.maxBHSpinMag
        if currJumpFac > massRangeParams.maxNSSpinMag:
            curr_spin_ns_jump_fac = massRangeParams.maxNSSpinMag
        spin1z = numpy.zeros(numJumpPoints, dtype=float)
        spin2z = numpy.zeros(numJumpPoints, dtype=float)
        split_point = int(numJumpPoints/2)
        # So set the first half to be at least within the BH range and the
        # second half to be at least within the NS range
        spin1z[:split_point] = bestSpin1z + \
                            ( (numpy.random.random(split_point) - 0.5)\
                              * curr_spin_bh_jump_fac)
        spin1z[split_point:] = bestSpin1z + \
                      ( (numpy.random.random(numJumpPoints-split_point) - 0.5)\
                        * curr_spin_ns_jump_fac)
        spin2z[:split_point] = bestSpin2z + \
                            ( (numpy.random.random(split_point) - 0.5)\
                              * curr_spin_bh_jump_fac)
        spin2z[split_point:] = bestSpin2z + \
                      ( (numpy.random.random(numJumpPoints-split_point) - 0.5)\
                        * curr_spin_ns_jump_fac)

    # Point[0] is always set to the original point
    chirpmass[0] = bestChirpmass
    eta[0] = bestEta
    spin1z[0] = bestSpin1z
    spin2z[0] = bestSpin2z

    # Remove points where eta becomes unphysical
    eta[eta > massRangeParams.maxEta] = massRangeParams.maxEta
    if massRangeParams.minEta:
        eta[eta < massRangeParams.minEta] = massRangeParams.minEta
    else:
        eta[eta < 0.0001] = 0.0001

    # Total mass, masses and mass diff
    totmass = chirpmass / (eta**(3./5.))
    diff = (totmass*totmass * (1-4*eta))**0.5
    mass1 = (totmass + diff)/2.
    mass2 = (totmass - diff)/2.

    # Check the validity of the spin values
    # Do the first spin

    if maxSpinMag == 0:
        # Shortcut if non-spinning
        pass
    elif massRangeParams.nsbhFlag or (maxSpinMag == minSpinMag):
        # Simple case where I don't have to worry about correlation with mass
        numploga = abs(spin1z) > massRangeParams.maxBHSpinMag
        spin1z[numploga] = 0
    else:
        # Do have to consider masses
        boundary_mass = massRangeParams.ns_bh_boundary_mass
        numploga1 = numpy.logical_and(mass1 >= boundary_mass,
                                   abs(spin1z) <= massRangeParams.maxBHSpinMag)
        numploga2 = numpy.logical_and(mass1 < boundary_mass,
                                   abs(spin1z) <= massRangeParams.maxNSSpinMag)
        numploga = numpy.logical_or(numploga1, numploga2)
        numploga = numpy.logical_not(numploga)
        spin1z[numploga] = 0

    # Same for the second spin

    if maxSpinMag == 0:
        # Shortcut if non-spinning
        pass
    elif massRangeParams.nsbhFlag or (maxSpinMag == minSpinMag):
        numplogb = abs(spin2z) > massRangeParams.maxNSSpinMag
        spin2z[numplogb] = 0
    else:
        # Do have to consider masses
        boundary_mass = massRangeParams.ns_bh_boundary_mass
        numplogb1 = numpy.logical_and(mass2 >= boundary_mass,
                                   abs(spin2z) <= massRangeParams.maxBHSpinMag)
        numplogb2 = numpy.logical_and(mass2 < boundary_mass,
                                   abs(spin2z) <= massRangeParams.maxNSSpinMag)
        numplogb = numpy.logical_or(numplogb1, numplogb2)
        numplogb = numpy.logical_not(numplogb)
        spin2z[numplogb] = 0

    if (maxSpinMag) and (numploga[0] or numplogb[0]):
        raise ValueError("Cannot remove the guide point!")

    # And remove points where the individual masses are outside of the physical
    # range. Or the total masses are.
    # These "removed" points will have metric distances that will be much, much
    # larger than any thresholds used in the functions in brute_force_utils.py
    # and will always be rejected. An unphysical value cannot be used as it
    # would result in unphysical metric distances and cause failures.
    totmass[mass1 < massRangeParams.minMass1*0.9999] = 0.0001
    totmass[mass1 > massRangeParams.maxMass1*1.0001] = 0.0001
    totmass[mass2 < massRangeParams.minMass2*0.9999] = 0.0001
    totmass[mass2 > massRangeParams.maxMass2*1.0001] = 0.0001
    # There is some numerical error which can push this a bit higher. We do
    # *not* want to reject the initial guide point. This error comes from
    # Masses -> totmass, eta -> masses conversion, we will have points pushing
    # onto the boudaries of the space.
    totmass[totmass > massRangeParams.maxTotMass*1.0001] = 0.0001
    totmass[totmass < massRangeParams.minTotMass*0.9999] = 0.0001
    if massRangeParams.max_chirp_mass:
        totmass[chirpmass > massRangeParams.max_chirp_mass*1.0001] = 0.0001
    if massRangeParams.min_chirp_mass:
        totmass[chirpmass < massRangeParams.min_chirp_mass*0.9999] = 0.0001

    if totmass[0] < 0.00011:
        raise ValueError("Cannot remove the guide point!")

    mass1[totmass < 0.00011] = 0.0001
    mass2[totmass < 0.00011] = 0.0001

    # Then map to xis
    new_xis = get_cov_params(mass1, mass2, spin1z, spin2z,
                             metricParams, fUpper)
    return totmass, eta, spin1z, spin2z, mass1, mass2, new_xis

def stack_xi_direction_brute(xis, bestMasses, bestXis, direction_num,
                             req_match, massRangeParams, metricParams, fUpper,
                             scaleFactor=0.8, numIterations=3000):
    """
    This function is used to assess the depth of the xi_space in a specified
    dimension at a specified point in the higher dimensions. It does this by
    iteratively throwing points at the space to find maxima and minima.

    Parameters
    -----------

    xis : list or array
        Position in the xi space at which to assess the depth. This can be only
        a subset of the higher dimensions than that being sampled.
    bestMasses : list
        Contains [totalMass, eta, spin1z, spin2z]. Is a physical position
        mapped to xi coordinates in bestXis that is close to the xis point.
        This is aimed to give the code a starting point.
    bestXis : list
        Contains the position of bestMasses in the xi coordinate system.
    direction_num : int
        The dimension that you want to assess the depth of (0 = 1, 1 = 2 ...)
    req_match : float
        When considering points to assess the depth with, only consider points
        with a mismatch that is smaller than this with xis.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    scaleFactor : float, optional (default = 0.8)
        The value of the scale factor that is used when calling
        pycbc.tmpltbank.get_mass_distribution.
    numIterations : int, optional (default = 3000)
        The number of times to make calls to get_mass_distribution when
        assessing the maximum/minimum of this parameter space. Making this
        smaller makes the code faster, but at the cost of accuracy.   
 
    Returns
    --------
    xi_min : float
        The minimal value of the specified dimension at the specified point in
        parameter space.
    xi_max : float
       The maximal value of the specified dimension at the specified point in
        parameter space.
    """

    # Find minimum
    ximin = find_xi_extrema_brute(xis, bestMasses, bestXis, direction_num, \
                                  req_match, massRangeParams, metricParams, \
                                  fUpper, find_minimum=True, \
                                  scaleFactor=scaleFactor, \
                                  numIterations=numIterations)
 
    # Find maximum
    ximax = find_xi_extrema_brute(xis, bestMasses, bestXis, direction_num, \
                                  req_match, massRangeParams, metricParams, \
                                  fUpper, find_minimum=False, \
                                  scaleFactor=scaleFactor, \
                                  numIterations=numIterations)

    return ximin, ximax

def find_xi_extrema_brute(xis, bestMasses, bestXis, direction_num, req_match, \
                          massRangeParams, metricParams, fUpper, \
                          find_minimum=False, scaleFactor=0.8, \
                          numIterations=3000):   
    """
    This function is used to find the largest or smallest value of the xi
    space in a specified
    dimension at a specified point in the higher dimensions. It does this by
    iteratively throwing points at the space to find extrema.

    Parameters
    -----------

    xis : list or array
        Position in the xi space at which to assess the depth. This can be only
        a subset of the higher dimensions than that being sampled.
    bestMasses : list
        Contains [totalMass, eta, spin1z, spin2z]. Is a physical position
        mapped to xi coordinates in bestXis that is close to the xis point.
        This is aimed to give the code a starting point.
    bestXis : list
        Contains the position of bestMasses in the xi coordinate system.
    direction_num : int
        The dimension that you want to assess the depth of (0 = 1, 1 = 2 ...)
    req_match : float
        When considering points to assess the depth with, only consider points
        with a mismatch that is smaller than this with xis.
    massRangeParams : massRangeParameters instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper that was used when obtaining the xi_i
        coordinates. This lets us know how to rotate potential physical points
        into the correct xi_i space. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)
    find_minimum : boolean, optional (default = False)
        If True, find the minimum value of the xi direction. If False find the
        maximum value.
    scaleFactor : float, optional (default = 0.8)
        The value of the scale factor that is used when calling
        pycbc.tmpltbank.get_mass_distribution.
    numIterations : int, optional (default = 3000)
        The number of times to make calls to get_mass_distribution when
        assessing the maximum/minimum of this parameter space. Making this
        smaller makes the code faster, but at the cost of accuracy.   

    Returns
    --------
    xi_extent : float
        The extremal value of the specified dimension at the specified point in
        parameter space.
    """

    # Setup
    xi_size = len(xis)
    bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
    if find_minimum:
        xiextrema = 10000000000
    else:
        xiextrema = -100000000000

    for _ in range(numIterations):
        # Evaluate extrema of the xi direction specified
        totmass, eta, spin1z, spin2z, _, _, new_xis = \
            get_mass_distribution([bestChirpmass,bestMasses[1],bestMasses[2],
                                   bestMasses[3]],
                                  scaleFactor, massRangeParams, metricParams,
                                  fUpper)
        cDist = (new_xis[0] - xis[0])**2
        for j in range(1, xi_size):
            cDist += (new_xis[j] - xis[j])**2
        redCDist = cDist[cDist < req_match]
        if len(redCDist):
            if not find_minimum:
                new_xis[direction_num][cDist > req_match] = -10000000
                currXiExtrema = (new_xis[direction_num]).max()
                idx = (new_xis[direction_num]).argmax()
            else:
                new_xis[direction_num][cDist > req_match] = 10000000
                currXiExtrema = (new_xis[direction_num]).min()
                idx = (new_xis[direction_num]).argmin()
            if ( ((not find_minimum) and (currXiExtrema > xiextrema)) or \
                         (find_minimum and (currXiExtrema < xiextrema)) ):
                xiextrema = currXiExtrema
                bestMasses[0] = totmass[idx]
                bestMasses[1] = eta[idx]
                bestMasses[2] = spin1z[idx]
                bestMasses[3] = spin2z[idx]
                bestChirpmass = bestMasses[0] * (bestMasses[1])**(3./5.)
    return xiextrema



# ---------------------------------------------------------------------------
# Deterministic derivative-based method (the default; see the module
# docstring). The public entry points are get_physical_covaried_masses_newton
# (inversion) and stack_xi_direction_continuation (depth). Everything else in
# this section is an internal helper.
# ---------------------------------------------------------------------------

# Central finite-difference steps for (m1, m2, s1z, s2z)
_FD_H = numpy.array([1e-5, 1e-5, 1e-6, 1e-6])


def _eval_pts(pts, metricParams, fUpper):
    """Evaluate the xi coordinates of pts (N, 4) -> array (n_xi, N)."""
    xis = get_cov_params(
        pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], metricParams, fUpper)
    return numpy.array(xis)


def _jac_and_center(x, metricParams, fUpper):
    """xi values and Jacobian at x via batched central differences.

    Returns (fx (n_xi,), J (n_xi, 4)) using a single vectorized map
    evaluation of 9 points.
    """
    pts = numpy.tile(x, (9, 1))
    for i in range(4):
        pts[1 + 2 * i, i] += _FD_H[i]
        pts[2 + 2 * i, i] -= _FD_H[i]
    xis = _eval_pts(pts, metricParams, fUpper)
    fx = xis[:, 0]
    J = numpy.empty((xis.shape[0], 4))
    for i in range(4):
        J[:, i] = (xis[:, 1 + 2 * i] - xis[:, 2 + 2 * i]) / (2 * _FD_H[i])
    return fx, J


def _spin_caps_scalar(m1, m2, mrp):
    """Per-component maximum |spin| for a single (m1, m2), applying the NS/BH
    boundary rule: below ns_bh_boundary_mass a component is capped at the NS
    spin, above it at the BH spin. Kept scalar (no array allocation) because
    it is on the hot path - hundreds of thousands of calls per bank.
    """
    maxS = mrp.maxNSSpinMag if mrp.maxNSSpinMag > mrp.maxBHSpinMag \
        else mrp.maxBHSpinMag
    minS = mrp.maxNSSpinMag if mrp.maxNSSpinMag < mrp.maxBHSpinMag \
        else mrp.maxBHSpinMag
    if mrp.nsbhFlag:
        return mrp.maxBHSpinMag, mrp.maxNSSpinMag
    if maxS == minS:
        return maxS, maxS
    b = mrp.ns_bh_boundary_mass
    cap1 = mrp.maxBHSpinMag if m1 >= b else mrp.maxNSSpinMag
    cap2 = mrp.maxBHSpinMag if m2 >= b else mrp.maxNSSpinMag
    return cap1, cap2


def _project(x, mrp):
    """Project a point onto the box/spin-cap constraints (m1 >= m2)."""
    m1, m2, s1, s2 = x
    if m1 < m2:
        m1, m2, s1, s2 = m2, m1, s2, s1
    m1 = min(max(m1, mrp.minMass1), mrp.maxMass1)
    m2 = min(max(m2, mrp.minMass2), mrp.maxMass2)
    if m2 > m1:
        m2 = m1
    cap1, cap2 = _spin_caps_scalar(m1, m2, mrp)
    s1 = min(max(s1, -cap1), cap1)
    s2 = min(max(s2, -cap2), cap2)
    return numpy.array([m1, m2, s1, s2])


def _is_valid(x, mrp):
    """Full physical validity check for a single point."""
    m1, m2, s1, s2 = x
    tol = 1e-6
    if m1 < m2 - tol:
        return False
    if m1 < mrp.minMass1 - tol or m1 > mrp.maxMass1 + tol:
        return False
    if m2 < mrp.minMass2 - tol or m2 > mrp.maxMass2 + tol:
        return False
    M = m1 + m2
    if M > mrp.maxTotMass + tol or M < mrp.minTotMass - tol:
        return False
    eta = m1 * m2 / (M * M)
    if eta > mrp.maxEta + 1e-9:
        return False
    min_eta = getattr(mrp, 'minEta', None)
    if min_eta and eta < min_eta - 1e-9:
        return False
    if mrp.max_chirp_mass or mrp.min_chirp_mass:
        mc = M * eta**0.6
        if mrp.max_chirp_mass and mc > mrp.max_chirp_mass * (1 + 1e-9):
            return False
        if mrp.min_chirp_mass and mc < mrp.min_chirp_mass * (1 - 1e-9):
            return False
    cap1, cap2 = _spin_caps_scalar(m1, m2, mrp)
    if abs(s1) > cap1 + 1e-9 or abs(s2) > cap2 + 1e-9:
        return False
    return True


def _seed_from_bestmasses(bestMasses, mrp):
    """Convert a [totmass, eta, s1z, s2z] seed to a projected (m1,m2,s1,s2)."""
    tot = float(bestMasses[0])
    eta = min(float(bestMasses[1]), 0.25)
    diff = numpy.sqrt(max(tot * tot * (1 - 4 * eta), 0.))
    m1 = (tot + diff) / 2.
    m2 = (tot - diff) / 2.
    return _project(
        numpy.array([m1, m2, float(bestMasses[2]), float(bestMasses[3])]), mrp)


def _gn_solve(target, x0, req_match, mrp, metricParams, fUpper, max_iter=25):
    """Damped Gauss-Newton solve of xi[:k](x) = target.

    Returns (x, fx, dist2, nfev, converged). Underdetermined steps use the
    min-norm least-squares solution; step lengths are chosen by evaluating a
    geometric ladder of candidates in one batched map call.
    """
    k = len(target)
    target = numpy.asarray(target, dtype=float)
    x = _project(numpy.asarray(x0, dtype=float), mrp)
    fx, J = _jac_and_center(x, metricParams, fUpper)
    nfev = 9
    r = fx[:k] - target
    d2 = float(r @ r)
    alphas = 2.0 ** -numpy.arange(7)
    for _ in range(max_iter):
        if d2 < req_match:
            return x, fx, d2, nfev, True
        dx = numpy.linalg.lstsq(J[:k], -r, rcond=None)[0]
        trials = numpy.array([_project(x + a * dx, mrp) for a in alphas])
        keep = numpy.array([_is_valid(t, mrp) for t in trials])
        if not keep.any():
            break
        tpts = trials[keep]
        txis = _eval_pts(tpts, metricParams, fUpper)
        nfev += len(tpts)
        tr = txis[:k] - target[:, None]
        td2 = numpy.einsum('ij,ij->j', tr, tr)
        j = int(td2.argmin())
        if td2[j] >= d2:
            break
        x = tpts[j]
        fx = txis[:, j]
        d2 = float(td2[j])
        if d2 < req_match:
            return x, fx, d2, nfev, True
        fx, J = _jac_and_center(x, metricParams, fUpper)
        nfev += 9
        r = fx[:k] - target
        d2 = float(r @ r)
    return x, fx, d2, nfev, d2 < req_match


def _alt_seeds(x0, mrp):
    """Deterministic alternative starting points for Gauss-Newton restarts."""
    seeds = []
    seeds.append(_project(numpy.array([x0[0], x0[1], 0., 0.]), mrp))
    for fac in (0.9, 1.1):
        seeds.append(_project(
            numpy.array([x0[0] * fac, x0[1] * fac, x0[2], x0[3]]), mrp))
    mmid1 = 0.5 * (mrp.minMass1 + mrp.maxMass1)
    mmid2 = 0.5 * (mrp.minMass2 + mrp.maxMass2)
    seeds.append(_project(numpy.array([mmid1, mmid2, x0[2], x0[3]]), mrp))
    return seeds


def get_physical_covaried_masses_newton(xis, bestMasses, bestXis, req_match,
                                        massRangeParams, metricParams, fUpper,
                                        giveUpThresh=500):
    """Deterministic Gauss-Newton inversion from the xi coordinate system to
    physical masses and spins.

    A drop-in replacement for :func:`get_physical_covaried_masses` with the
    same call and return signature. Rather than the stochastic jump search it
    solves ``xi(m1, m2, s1z, s2z) = xis`` by damped Gauss-Newton on a batched
    central-difference Jacobian, starting from ``bestMasses``. If that stalls
    it retries from a few deterministic alternative seeds, and only as a last
    resort falls back to :func:`get_physical_covaried_masses` (seeded from the
    target so the result is reproducible). The whole routine is deterministic.

    Parameters
    -----------
    xis : list or array
        The target position in the xi coordinate system. Only the first
        ``len(xis)`` dimensions are matched.
    bestMasses : list
        [totalMass, eta, spin1z, spin2z] of a physical point close to xis,
        used as the starting point.
    bestXis : list
        The xi coordinates of bestMasses. Accepted for signature
        compatibility with the legacy method and passed to the fallback; not
        otherwise required here.
    req_match : float
        Convergence tolerance: iteration stops once the squared xi distance to
        the target is below this value.
    massRangeParams : massRangeParameters instance
        Mass and spin range limits defining the physical region.
    metricParams : metricParameters instance
        Structure holding the metric eigenvalues, eigenvectors and covariance
        matrix needed to move between physical and xi coordinates.
    fUpper : float
        The upper frequency cutoff used when obtaining the xi coordinates.
        Must be a key in metricParams.evals, metricParams.evecs and
        metricParams.evecsCV.
    giveUpThresh : int, optional
        Iteration budget handed to the legacy fallback search.

    Returns
    --------
    mass1 : float
        Recovered mass of the heavier body.
    mass2 : float
        Recovered mass of the lighter body.
    spin1z : float
        Recovered spin of the heavier body.
    spin2z : float
        Recovered spin of the lighter body.
    count : int
        Number of map evaluations used.
    mismatch : float
        Squared xi distance between the recovered point and xis.
    new_xis : list
        The xi coordinates of the recovered point.
    """
    x0 = _seed_from_bestmasses(bestMasses, massRangeParams)
    x, fx, d2, nfev, ok = _gn_solve(xis, x0, req_match, massRangeParams,
                                    metricParams, fUpper)
    if not ok:
        for xalt in _alt_seeds(x0, massRangeParams):
            x2, fx2, d22, n2, ok = _gn_solve(xis, xalt, req_match,
                                             massRangeParams, metricParams,
                                             fUpper)
            nfev += n2
            if d22 < d2:
                x, fx, d2 = x2, fx2, d22
            if ok:
                break
    if ok:
        return x[0], x[1], x[2], x[3], nfev, d2, fx
    logger.info("Gauss-Newton stalled (dist^2=%.3e); falling back to the "
            "legacy brute-force search", d2)
    # Seed numpy's global RNG from the target coordinates (and restore it
    # afterwards) so the fallback - and therefore the whole newton method -
    # is deterministic. Note hashes of number tuples do not depend on
    # PYTHONHASHSEED, so this is stable across processes.
    seed = abs(hash(tuple(numpy.round(numpy.asarray(xis, dtype=float), 10)))) \
        % (2**32)
    state = numpy.random.get_state()
    numpy.random.seed(seed)
    try:
        result = get_physical_covaried_masses(
            xis, bestMasses, bestXis, req_match, massRangeParams,
            metricParams, fUpper, giveUpThresh=giveUpThresh)
    finally:
        numpy.random.set_state(state)
    if result[5] < d2:
        return result
    return x[0], x[1], x[2], x[3], nfev, d2, fx


def _correct_fixedJ(target, x, Bpinv, req_match, mrp, metricParams, fUpper,
                    max_it=6):
    """Corrector: pull x back into the req_match ball using a *fixed*
    Jacobian pseudo-inverse Bpinv (from the predictor point). Costs one map
    evaluation per iteration instead of re-differentiating, which is valid
    because the predictor step is small so the Jacobian barely changes.

    Returns (x, fx, d2, nfev, in_ball).
    """
    k = len(target)
    nfev = 0
    fx = _eval_pts(x[None, :], metricParams, fUpper)[:, 0]
    nfev += 1
    r = fx[:k] - target
    d2 = float(r @ r)
    for _ in range(max_it):
        if d2 < req_match:
            return x, fx, d2, nfev, True
        dx = -(Bpinv @ r)
        # Damped: try the full step first (the common case), backing off only
        # if the fixed Jacobian overshoots near a boundary.
        for alpha in (1.0, 0.5, 0.25):
            xt = _project(x + alpha * dx, mrp)
            ft = _eval_pts(xt[None, :], metricParams, fUpper)[:, 0]
            nfev += 1
            rt = ft[:k] - target
            dt2 = float(rt @ rt)
            if dt2 < d2:
                x, fx, r, d2 = xt, ft, rt, dt2
                break
        else:
            break
    return x, fx, d2, nfev, d2 < req_match


def _continuation_extremum(target, x0, fx0, dnum, sign, req_match, mrp,
                           metricParams, fUpper, max_steps=80):
    """Predictor-corrector walk of xi[dnum] to its extremum on the sheet
    {xi[:k] == target}, staying inside the req_match ball via the corrector.

    Predictor: step along the steepest-ascent direction of xi[dnum] projected
    into the null space of the constraint Jacobian (so xi[:k] is unchanged to
    first order). Corrector: a few fixed-Jacobian Newton steps back into the
    req_match ball, reusing the pseudo-inverse computed for the predictor.
    The step is grown on progress and backtracked otherwise; the walk
    terminates when no feasible ascent step improves xi[dnum] (typically at a
    physical boundary).

    Returns (best_val, best_x, nfev).
    """
    k = len(target)
    x = x0.copy()
    best_val = float(fx0[dnum])
    best_x = x.copy()
    step = 0.3
    min_step = 1e-4
    nfev = 0
    eye = numpy.eye(4)

    def _direction(x):
        # Jacobian, ascent direction in the sheet tangent, and the pseudo-
        # inverse the corrector reuses. Returns None if there is no ascent.
        fx, J = _jac_and_center(x, metricParams, fUpper)
        B = J[:k]
        Bpinv = numpy.linalg.pinv(B)
        d = sign * ((eye - Bpinv @ B) @ J[dnum])
        nd = numpy.linalg.norm(d)
        if nd < 1e-10:
            return None
        return d / nd, Bpinv

    got = _direction(x)
    nfev += 9
    if got is None:
        return best_val, best_x, nfev
    d, Bpinv = got
    for _ in range(max_steps):
        improved = False
        for frac in (1.0, 0.35, 0.12):
            x_pred = _project(x + step * frac * d, mrp)
            x_new, fx_new, d2, nf, ok = _correct_fixedJ(
                target, x_pred, Bpinv, req_match, mrp, metricParams, fUpper)
            nfev += nf
            if ok and sign * (fx_new[dnum] - best_val) > 1e-6:
                x = x_new
                best_val = float(fx_new[dnum])
                best_x = x_new
                step = min(step * frac * 1.5, 4.0)
                improved = True
                break
        if improved:
            # x moved: refresh the ascent direction and Jacobian at the new x
            got = _direction(x)
            nfev += 9
            if got is None:
                break
            d, Bpinv = got
        else:
            # x unchanged: only backtrack the step; keep d and Bpinv
            step *= 0.3
            if step < min_step:
                break
    return best_val, best_x, nfev


def stack_xi_direction_continuation(xis, bestMasses, bestXis, direction_num,
                                    req_match, massRangeParams, metricParams,
                                    fUpper, **kwargs):
    """Deterministic depth measurement in a specified xi direction.

    A drop-in replacement for :func:`stack_xi_direction_brute`. It finds the
    smallest and largest value of ``xi[direction_num]`` over the physically
    valid points whose lower xi coordinates lie within the req_match ball of
    the target, by walking the feasible surface with a predictor-corrector
    continuation (see :func:`_continuation_extremum`) rather than throwing
    random points at it. The result is deterministic.

    Parameters
    -----------
    xis : list or array
        Position in the xi space at which to assess the depth. May give only
        the lower dimensions that are held fixed while direction_num is varied.
    bestMasses : list
        [totalMass, eta, spin1z, spin2z] of a physical point close to xis,
        used as the starting point.
    bestXis : list
        The xi coordinates of bestMasses. Accepted for signature
        compatibility with the legacy method; not otherwise required here.
    direction_num : int
        The xi dimension whose extent (depth) is measured (0 = xi_1, ...).
    req_match : float
        Only points within this squared xi distance of xis are considered.
    massRangeParams : massRangeParameters instance
        Mass and spin range limits defining the physical region.
    metricParams : metricParameters instance
        Structure holding the metric eigenvalues, eigenvectors and covariance
        matrix needed to move between physical and xi coordinates.
    fUpper : float
        The upper frequency cutoff used when obtaining the xi coordinates.
        Must be a key in metricParams.evals, metricParams.evecs and
        metricParams.evecsCV.

    Returns
    --------
    xi_min : float
        The minimal value of xi[direction_num] over the valid region, or the
        sentinel 1e10 if no feasible starting point can be found.
    xi_max : float
        The maximal value of xi[direction_num] over the valid region, or the
        sentinel -1e10 if no feasible starting point can be found.
    """
    target = numpy.asarray(xis, dtype=float)
    k = len(target)
    x0 = _seed_from_bestmasses(bestMasses, massRangeParams)
    fx0 = _eval_pts(x0[None, :], metricParams, fUpper)[:, 0]
    r = fx0[:k] - target
    if float(r @ r) > req_match or not _is_valid(x0, massRangeParams):
        # Seed is not in the ball; solve for a feasible starting point
        x0, fx0, d2, _, ok = _gn_solve(target, x0, req_match, massRangeParams,
                                       metricParams, fUpper)
        if not ok:
            return 1e10, -1e10
    ximin, _, _ = _continuation_extremum(target, x0, fx0, direction_num, -1.,
                                         req_match, massRangeParams,
                                         metricParams, fUpper)
    ximax, _, _ = _continuation_extremum(target, x0, fx0, direction_num, +1.,
                                         req_match, massRangeParams,
                                         metricParams, fUpper)
    return ximin, ximax

