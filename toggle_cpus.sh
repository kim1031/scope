#!/bin/bash

switch=$1

for cpuid in ${@:2}
do
    echo $switch | sudo tee /sys/devices/system/cpu/cpu$cpuid/online &
done

wait
