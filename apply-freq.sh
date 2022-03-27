#!/bin/bash

SYSTEM_CPU_DIR="/sys/devices/system/cpu"

set_governors() {
    for i in $(seq 0 $(getconf _NPROCESSORS_ONLN));
    do
        echo performance > /sys/devices/system/cpu/cpu${i}/cpufreq/scaling_governor
    done
}

apply_locally() {
    set_governors

    # Reset frequencies
    echo 100 > /sys/devices/system/cpu/intel_pstate/max_perf_pct
    echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct

    SYSTEM_CPU_MAX_FREQ="$(cat "${SYSTEM_CPU_DIR}/cpu0/cpufreq/cpuinfo_max_freq")"
    SYSTEM_CPU_MIN_FREQ="$(cat "${SYSTEM_CPU_DIR}/cpu0/cpufreq/cpuinfo_min_freq")"
    target_pct=$(( $1 * 100 / SYSTEM_CPU_MAX_FREQ))

    echo ${target_pct} > /sys/devices/system/cpu/intel_pstate/min_perf_pct

    echo ${target_pct} > /sys/devices/system/cpu/intel_pstate/max_perf_pct
}

apply_remotely() {
    scp ./apply-freq.sh cc@${1}:/tmp/apply-freq.sh
    ssh cc@$1 "sudo bash /tmp/apply-freq.sh $2"
}

if [ -z $2 ];
then
    apply_locally $1

else
    apply_remotely $2 $1
fi
