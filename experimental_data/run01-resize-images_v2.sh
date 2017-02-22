#!/bin/bash

##sizx=128
##sizy=128
##sizz=64

##sizx=64
##sizy=64
##sizz=32

sizx=256
sizy=256
sizz=64

##sizx=512
##sizy=512
##sizz=64

newSize="${sizx}x${sizy}x${sizz}"

idir="${PWD}/original"
odir="${PWD}/resize-${newSize}"

mkdir -p "${odir}"

ls -1 ${idir}/*[0-9].nii.gz | while read finp
do
    finpMsk="${finp}-segmbT.nii.gz"
    bnInp=`basename ${finp} .nii.gz`
    fout="${odir}/${bnInp}-${newSize}.nii.gz"
    foutMsk="${odir}/${bnInp}-${newSize}-msk.nii.gz"
    echo ":: process [${bnInp}]"
    echo -e "\t${finp} -> ${fout}"
    c3d ${finp} -interpolation NearestNeighbor -resample ${newSize} -o ${fout}
##    c3d ${finp} -interpolation Cubic -resample ${newSize} -o ${fout}
    echo -e "\t${finpMsk} -> ${foutMsk}"
    c3d ${finpMsk} -interpolation NearestNeighbor -resample ${newSize} -o ${foutMsk}
##    c3d ${finpMsk} -interpolation Cubic -resample ${newSize} -o ${foutMsk}
##    exit 1
done
