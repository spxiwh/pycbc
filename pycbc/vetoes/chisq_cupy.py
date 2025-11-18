# Copyright (C) 2015  Alex Nitz, Josh Willis
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

import functools
import numpy
import cupy as cp
import lal
from mako.template import Template

LALARGS = {
    'TWOPI': lal.TWOPI,
}

accum_diff_sq_kernel = cp.ElementwiseKernel(
    "X input",
    "raw Y output",
    "output[i] += norm(input)",
    "accum_diff_sq_kernel"
)

def chisq_accum_bin(chisq, q):
    accum_diff_sq_kernel(q.data, chisq.data)


chisqkernel = Template("""
#include <cstdint>
extern "C" __global__ void power_chisq_at_points_${NP}(
                                      %if fuse:
                                          float2* htilde,
                                          float2* stilde,
                                      %else:
                                          float2* corr,
                                      %endif
                                      float2* outc, unsigned int N,
                                      %for p in range(NP):
                                        float phase${p},
                                      %endfor
                                      uint32_t* kmin,
                                      uint32_t* kmax,
                                      uint32_t* bv,
                                      unsigned int nbins){
    __shared__ unsigned int s;
    __shared__ unsigned int e;
    __shared__ float2 chisq[${NT} * ${NP}];

    // load integration boundaries (might not be bin boundaries if bin is large)
    if (threadIdx.x == 0){
        s = kmin[blockIdx.x];
        e = kmax[blockIdx.x];
    }

    % for p in range(NP):
        chisq[threadIdx.x + ${NT*p}].x = 0;
        chisq[threadIdx.x + ${NT*p}].y = 0;
    % endfor
    __syncthreads();

    // calculate the chisq integral for each thread
    // sliding reduction for each thread from s, e
    for (int i = threadIdx.x + s; i < e; i += blockDim.x){
        float re, im;

        %if fuse:
            float2 qt, st, ht;
            st = stilde[i];
            ht = htilde[i];
            qt.x = ht.x * st.x + ht.y * st.y;
            qt.y = ht.x * st.y - ht.y * st.x;
        %else:
            float2 qt = corr[i];
        %endif

        %for p in range(NP):
            sincosf(phase${p} * i, &im, &re);
            chisq[threadIdx.x + ${NT*p}].x += re * qt.x - im * qt.y;
            chisq[threadIdx.x + ${NT*p}].y += im * qt.x + re * qt.y;
        %endfor
    }

    float x, y, x2, y2;
    // logarithmic reduction within thread block
    for (int j=${NT} / 2; j>=1; j/=2){
        if (threadIdx.x <j){
            %for p in range(NP):
                __syncthreads();
                x = chisq[threadIdx.x + ${NT*p}].x;
                y = chisq[threadIdx.x + ${NT*p}].y;
                x2 = chisq[threadIdx.x + j + ${NT*p}].x;
                y2 = chisq[threadIdx.x + j + ${NT*p}].y;
                 __syncthreads();
                chisq[threadIdx.x + ${NT*p}].x = x + x2;
                chisq[threadIdx.x + ${NT*p}].y = y + y2;
            %endfor
        }
    }

    if (threadIdx.x == 0){
        % for p in range(NP):
            atomicAdd(&outc[bv[blockIdx.x] + nbins * ${p}].x, chisq[0 + ${NT*p}].x);
            atomicAdd(&outc[bv[blockIdx.x] + nbins * ${p}].y, chisq[0 + ${NT*p}].y);
        % endfor
    }

}
""")

chisqkernel_pow2 = Template("""
#include <cstdint>
extern "C" __global__ void power_chisq_at_points_${NP}_pow2(
                                      %if fuse:
                                          float2* htilde,
                                          float2* stilde,
                                      %else:
                                          float2* corr,
                                      %endif
                                      float2* outc, unsigned int N,
                                      %for p in range(NP):
                                        unsigned int points${p},
                                      %endfor
                                      uint32_t* kmin,
                                      uint32_t* kmax,
                                      uint32_t* bv,
                                      unsigned int nbins){
    __shared__ unsigned int s;
    __shared__ unsigned int e;
    __shared__ float2 chisq[${NT} * ${NP}];
    float twopi = ${TWOPI};
    unsigned long long NN;

    NN = (unsigned long long) N;

    // load integration boundaries (might not be bin boundaries if bin is large)
    if (threadIdx.x == 0){
        s = kmin[blockIdx.x];
        e = kmax[blockIdx.x];
    }

    % for p in range(NP):
        chisq[threadIdx.x + ${NT*p}].x = 0;
        chisq[threadIdx.x + ${NT*p}].y = 0;
    % endfor
    __syncthreads();

    // calculate the chisq integral for each thread
    // sliding reduction for each thread from s, e
    for (int i = threadIdx.x + s; i < e; i += blockDim.x){
        float re, im;

        %if fuse:
            float2 qt, st, ht;
            st = stilde[i];
            ht = htilde[i];
            qt.x = ht.x * st.x + ht.y * st.y;
            qt.y = ht.x * st.y - ht.y * st.x;
        %else:
            float2 qt = corr[i];
        %endif

        %for p in range(NP):
            unsigned long long prod${p} = points${p} * i;
            unsigned int k${p} = (unsigned int) (prod${p}&(NN-1));
            float phase${p} = twopi * k${p}/((float) N);
            __sincosf(phase${p}, &im, &re);
            chisq[threadIdx.x + ${NT*p}].x += re * qt.x - im * qt.y;
            chisq[threadIdx.x + ${NT*p}].y += im * qt.x + re * qt.y;
        %endfor
    }

    float x, y, x2, y2;
    // logarithmic reduction within thread block
    for (int j=${NT} / 2; j>=1; j/=2){
        if (threadIdx.x <j){
            %for p in range(NP):
                __syncthreads();
                x = chisq[threadIdx.x + ${NT*p}].x;
                y = chisq[threadIdx.x + ${NT*p}].y;
                x2 = chisq[threadIdx.x + j + ${NT*p}].x;
                y2 = chisq[threadIdx.x + j + ${NT*p}].y;
                 __syncthreads();
                chisq[threadIdx.x + ${NT*p}].x = x + x2;
                chisq[threadIdx.x + ${NT*p}].y = y + y2;
            %endfor
        }
    }

    if (threadIdx.x == 0){
        % for p in range(NP):
            atomicAdd(&outc[bv[blockIdx.x] + nbins * ${p}].x, chisq[0 + ${NT*p}].x);
            atomicAdd(&outc[bv[blockIdx.x] + nbins * ${p}].y, chisq[0 + ${NT*p}].y);
        % endfor
    }

}
""")


chisqkernel_pow2 = Template("""
#include <cstdint>
extern "C" __global__ void power_chisq_at_points_pow2_batch(
    float2* corr, // 2D: num_templates X N
    float2* outc, // 1D: num_points X max_bin_length 
    unsigned int N, // Scalar
    uint32_t* points, // 1D: num_points
    uint32_t* kmin, // 2D: num_templates X max_bin_size
    uint32_t* kmax, // 2D: num_templates X max_bin_size
    uint32_t* bv, // 2D: num_templates X max_bin_size
    uint32_t* nbins, // 1D: num_templates
    uint32_t* mapping, // 1D: num_points
    uint32_t* nb, // 1D: num_templates
    unsigned int num_points, // scalar
    unsigned int num_templates // scalar
)
{
    __shared__ unsigned int s;
    __shared__ unsigned int e;
    __shared__ unsigned int pnum;
    __shared__ unsigned int binnum;
    __shared__ unsigned int tempnum;
    __shared__ float2 chisq[${NT} * num_points];
    float twopi = ${TWOPI};
    unsigned long long NN;

    NN = (unsigned long long) N;

    // load integration boundaries (might not be bin boundaries if bin is large)
    if (threadIdx.x == 0){
        pnum = blockIdx.y;
        binnum = blockIdx.x;
        tempnum = mapping[pnum];
        // No work needed in this case
        if (binnum > nbins[tempnum])
        { 
            e = 0;
            s = 0;
        }
        else
        {
            s = kmin[blockIdx.x];
            e = kmax[blockIdx.x];
        }
    }

    chisq[threadIdx.x].x = 0;
    chisq[threadIdx.x].y = 0;
    __syncthreads();

    // GOT AS FAR AS HERE IN EXPANDING THIS TO BATCHED MODE

    // calculate the chisq integral for each thread
    // sliding reduction for each thread from s, e
    for (int i = threadIdx.x + s; i < e; i += blockDim.x){
        float re, im;

        %if fuse:
            float2 qt, st, ht;
            st = stilde[i];
            ht = htilde[i];
            qt.x = ht.x * st.x + ht.y * st.y;
            qt.y = ht.x * st.y - ht.y * st.x;
        %else:
            float2 qt = corr[i];
        %endif

        %for p in range(NP):
            unsigned long long prod${p} = points${p} * i;
            unsigned int k${p} = (unsigned int) (prod${p}&(NN-1));
            float phase${p} = twopi * k${p}/((float) N);
            __sincosf(phase${p}, &im, &re);
            chisq[threadIdx.x + ${NT*p}].x += re * qt.x - im * qt.y;
            chisq[threadIdx.x + ${NT*p}].y += im * qt.x + re * qt.y;
        %endfor
    }

    float x, y, x2, y2;
    // logarithmic reduction within thread block
    for (int j=${NT} / 2; j>=1; j/=2){
        if (threadIdx.x <j){
            %for p in range(NP):
                __syncthreads();
                x = chisq[threadIdx.x + ${NT*p}].x;
                y = chisq[threadIdx.x + ${NT*p}].y;
                x2 = chisq[threadIdx.x + j + ${NT*p}].x;
                y2 = chisq[threadIdx.x + j + ${NT*p}].y;
                 __syncthreads();
                chisq[threadIdx.x + ${NT*p}].x = x + x2;
                chisq[threadIdx.x + ${NT*p}].y = y + y2;
            %endfor
        }
    }

    if (threadIdx.x == 0){
        % for p in range(NP):
            atomicAdd(&outc[bv[blockIdx.x] + nbins * ${p}].x, chisq[0 + ${NT*p}].x);
            atomicAdd(&outc[bv[blockIdx.x] + nbins * ${p}].y, chisq[0 + ${NT*p}].y);
        % endfor
    }

}
""")


@functools.lru_cache(maxsize=None)
def get_pchisq_fn(np, fuse_correlate=False):
    nt = 256
    fn = cp.RawKernel(
        chisqkernel.render(NT=nt, NP=np, fuse=fuse_correlate, **LALARGS),
        f'power_chisq_at_points_{np}',
        backend='nvcc'
    )
    return fn, nt


@functools.lru_cache(maxsize=None)
def get_pchisq_fn_pow2(np, fuse_correlate=False):
    nt = 256
    fn = cp.RawKernel(
        chisqkernel_pow2.render(NT=nt, NP=np, fuse=fuse_correlate, **LALARGS),
        f'power_chisq_at_points_{np}_pow2',
        backend='nvcc'
    )
    return fn, nt


@functools.lru_cache(maxsize=None)
def get_pchisq_fn_pow2_batch():
    nt = 256
    fn = cp.RawKernel(
        chisqkernel_pow2.render(NT=nt, **LALARGS),
        f'power_chisq_at_points_pow2_batch',
        backend='nvcc'
    )
    return fn, nt


def get_cached_bin_layout(bins):
    bv, kmin, kmax = [], [], []
    for i in range(len(bins)-1):
        s, e = bins[i], bins[i+1]
        BS = 4096
        if (e - s) < BS:
            bv.append(i)
            kmin.append(s)
            kmax.append(e)
        else:
            k = list(numpy.arange(s, e, BS/2))
            kmin += k
            kmax += k[1:] + [e]
            bv += [i]*len(k)
    bv = cp.array(bv, dtype=cp.uint32)
    kmin = cp.array(kmin, dtype=cp.uint32)
    kmax = cp.array(kmax, dtype=cp.uint32)
    return kmin, kmax, bv

def get_cached_bin_layout_batch(bins, bv, kmin, kmax, nb):
    # FIXME: This is probably slow :-(
    for idx1, cbins in enumerate(bins):
        idx2 = 0
        for i in range(len(cbins)-1):
            s, e = cbins[i], cbins[i+1]
            BS = 4096
            if (e - s) < BS:
                bv[idx1,idx2] = i
                kmin[idx1,idx2] = s
                kmax[idx1,idx2] = e
                idx2 +=1
            else:
                k = cp.arange(s, e, BS/2)
                kmin[idx1,idx2:idx2+len(k)] = k
                kmax[idx1,idx2:idx2+len(k)-1] = k[1:]
                kmax[idx1,idx2+len(k)] = e
                bv[idx1,idx2:idx2+len(k)] = i
        nb[idx1] = idx2+1


def shift_sum_points(num, N, arg_tuple):
    #fuse = 'fuse' in corr.gpu_callback_method
    fuse = False

    fn, nt = get_pchisq_fn(num, fuse_correlate = fuse)
    corr, outp, phase, np, nb, N, kmin, kmax, bv, nbins = arg_tuple
    if fuse:
        args = [corr.htilde.data, corr.stilde.data]
    else:
        args = [corr.data]
    args += [outp, N] + phase[0:num]
    args += [kmin, kmax, bv, nbins]
    fn(
        (nb,),
        (nt,),
        *args,
    )
    
    outp = outp[num*nbins:]
    phase = phase[num:]
    np -= num
    return outp, phase, np

def shift_sum_points_pow2(num, arg_tuple):
    #fuse = 'fuse' in corr.gpu_callback_method
    fuse = False

    fn, nt = get_pchisq_fn_pow2(num, fuse_correlate = fuse)

    corr, outp, points, np, nb, N, kmin, kmax, bv, nbins = arg_tuple
    if fuse:
        args = [corr.htilde.data, corr.stilde.data]
    else:
        args = [corr.data]
    args += [outp, N] + points[0:num] + [kmin, kmax, bv, nbins]
    fn(
        (nb,),
        (nt,),
        tuple(args)
    )
            
    outp = outp[num*nbins:]
    points = points[num:]
    np -= num
    return outp, points, np

def shift_sum_points_pow2_batch(arg_tuple):
    fn, nt = get_pchisq_fn_pow2_batch()

    corr, outp, points, np, nb, N, kmin, kmax, bv, nbins, mapping = arg_tuple
    args = (corr, outp, N, points, kmin, kmax, bv, nbins, mapping, nb)

    fn(
        (max(nb),len(nb)),
        (nt,),
        args
    )


@functools.lru_cache(maxsize=None)
def get_cached_pow2(N):
    return not(N & (N-1))

def shift_sum(corr, points, bins):
    kmin, kmax, bv = get_cached_bin_layout(bins)
    nb = len(kmin)
    N = cp.uint32(len(corr))
    is_pow2 = get_cached_pow2(N)
    nbins = cp.uint32(len(bins) - 1)
    outc = cp.zeros((len(points), nbins), dtype=cp.complex64)
    outp = outc.reshape(nbins * len(points))
    np = len(points)

    if is_pow2:
        lpoints = points.tolist()
        while np > 0:
            cargs = (corr, outp, lpoints, np, nb, N, kmin, kmax, bv, nbins)

            if np >= 4:
                outp, lpoints, np = shift_sum_points_pow2(4, cargs)
            elif np >= 3:
                outp, lpoints, np = shift_sum_points_pow2(3, cargs)
            elif np >= 2:
                outp, lpoints, np = shift_sum_points_pow2(2, cargs)
            elif np == 1:
                outp, lpoints, np = shift_sum_points_pow2(1, cargs)
    else:
        phase = [cp.float32(p * 2.0 * cp.pi / N) for p in points]
        while np > 0:
            cargs = (corr, outp, phase, np, nb, N, kmin, kmax, bv, nbins)

            if np >= 4:
                outp, phase, np = shift_sum_points(4, cargs) # pylint:disable=no-value-for-parameter
            elif np >= 3:
                outp, phase, np = shift_sum_points(3, cargs) # pylint:disable=no-value-for-parameter
            elif np >= 2:
                outp, phase, np = shift_sum_points(2, cargs) # pylint:disable=no-value-for-parameter
            elif np == 1:
                outp, phase, np = shift_sum_points(1, cargs) # pylint:disable=no-value-for-parameter

    return (outc.conj() * outc).sum(axis=1).real

def shift_sum_batch(corr, points, bins, bin_lengths, mapping):
    # The bin layout can return longer lengths than the bin lengths.
    # Assuming a factor of 4 is enough(??). It will fail if not.
    max_bin_size = cp.uint32(bin_lengths.max().get()) * 4
    bv = cp.zeros([len(bins), max_bin_size], dtype=cp.uint32)
    kmin = cp.zeros([len(bins), max_bin_size], dtype=cp.uint32)
    kmax = cp.zeros([len(bins), max_bin_size], dtype=cp.uint32)
    nb = cp.zeros(len(bins), dtype=cp.uint32)

    get_cached_bin_layout_batch(bins, bv, kmin, kmax, nb)
    N = cp.uint32(len(corr[0]))
    is_pow2 = get_cached_pow2(N)
    nbins = bin_lengths - 1
    outc = cp.zeros((len(points), max_bin_size//4), dtype=cp.complex64)
    outp = outc.reshape((max_bin_size//4) * len(points))
    np = len(points)

    if is_pow2:
        cargs = (corr, outp, points, np, nb, N, kmin, kmax, bv, nbins, mapping)
        shift_sum_points_pow2_batch(cargs)
    else:
        raise NotImplementedError()
        phase = [cp.float32(p * 2.0 * cp.pi / N) for p in points]
        cargs = (corr, outp, phase, np, nb, N, kmin, kmax, bv, nbins, mapping)
        shift_sum_points_batch(cargs)

    return (outc.conj() * outc).sum(axis=1).real

