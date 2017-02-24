#!/bin/bash

fidx='idx.txt'

echo "path,pathmsk" > $fidx
for ii in `ls -1 *[0-9].nii.gz | sort -n`
do
    tbn=`basename $ii .nii.gz`
    tmsk="${tbn}-msk.nii.gz"
    echo "${ii},${tmsk}" | tee -a $fidx
done
