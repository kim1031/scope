#!/bin/bash

apply_locally() {
    # Reset frequencies
    sudo wrmsr 0x700 0x2000000000000000
    sudo wrmsr 0x703 0x00400000

    for cpuid in $(seq 0 24)
    do
        sudo wrmsr -p $cpuid 0x620 $1 &
    done
    wait
}

apply_remotely() {
    scp ./apply-uncore.sh cc@${1}:/tmp/apply-uncore.sh
    ssh cc@$1 "sudo bash /tmp/apply-uncore.sh $2"
}

if [ -z $2 ];
then
    apply_locally $1

else
    apply_remotely $2 $1
fi
