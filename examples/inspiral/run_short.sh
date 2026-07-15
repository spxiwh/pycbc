#!/bin/bash -e

function inspiral_run {
echo -e "\\n\\n>> [`date`] Running pycbc inspiral $1:$3 with $2 threads"
# Uncomment if you want profile information
#python -m cProfile -o output_$1_$2_$3.pstats
`which pycbc_inspiral` \
--frame-files DATA_FILE.gwf \
--sample-rate 2048 \
--sgchisq-snr-threshold 6.0 \
--sgchisq-locations "mtotal>40:20-30,20-45,20-60,20-75,20-90,20-105,20-120" \
--segment-end-pad 16 \
--low-frequency-cutoff 30 \
--pad-data 8 \
--cluster-window 1 \
--cluster-function symmetric \
--injection-window 4.5 \
--segment-start-pad 112 \
--psd-segment-stride 8 \
--psd-inverse-length 16 \
--filter-inj-only \
--psd-segment-length 16 \
--snr-threshold 5.5 \
--segment-length 256 \
--autogating-width 0.25 \
--autogating-threshold 100 \
--autogating-cluster 0.5 \
--autogating-taper 0.25 \
--newsnr-threshold 5 \
--psd-estimation median \
--strain-high-pass 20 \
--order -1 \
--chisq-bins "1.75*(get_freq('fSEOBNRv2Peak',params.mass1,params.mass2,params.spin1z,params.spin2z)-60.)**0.5" \
--channel-name H1:LOSC-STRAIN \
--gps-start-time 1126259078 \
--gps-end-time 1126259846 \
--output H1-INSPIRAL_$1_$2_$3-OUT.hdf \
--approximant "SPAtmplt:mtotal<4" "IMRPhenomD:else" \
--fft-backends $1 \
--processing-scheme cpu:$2 \
--fftw-threads-backend $3 \
--use-compressed-waveforms \
--bank-file REDUCED_BANK_GW150914.hdf \
--psdvar-freq-dependent \
--psdvar-segment 8 \
--psdvar-short-segment 0.25 \
--psdvar-long-segment 512 \
--psdvar-psd-duration 8 \
--psdvar-psd-stride 4 \
--psdvar-low-freq 20 \
--psdvar-high-freq 480 \
--psdvar-threshold 1.1 \
--verbose
#--verbose 2> inspiral_$1_$2_$3.log
# Uncomment above if you want logging

# Uncomment for profile pngs
#gprof2dot -f pstats output_$1_$2_$3.pstats | dot -Tpng -o $1_$2_$3.png

echo -e "\\n\\n>> [`date`] test for GW150914"
python ./check_GW150914_detection.py H1-INSPIRAL_$1_$2_$3-OUT.hdf
}

# Test pycbc inspiral by running over GW150914 with a limited template bank
echo -e "\\n\\n>> [`date`] Getting template bank"
wget -nv -nc https://github.com/gwastro/pycbc-config/raw/master/test/inspiral/SMALLER_BANK_FOR_GW150914.hdf

echo -e "\\n\\n>> [`date`] Creating data file"
pycbc_condition_strain \
    --frame-type GWOSC \
    --sample-rate 2048 \
    --pad-data 8 \
    --autogating-width 0.25 \
    --autogating-threshold 100 \
    --autogating-cluster 0.5 \
    --autogating-taper 0.25 \
    --strain-high-pass 10 \
    --channel-name H1:LOSC-STRAIN \
    --gps-start-time 1126258578 \
    --gps-end-time 1126259946 \
    --output-strain-file DATA_FILE.gwf \


start=`date +%s`
inspiral_run fftw 16 openmp
end=`date +%s`
runtime_fftw_openmp_16=$((end-start))

