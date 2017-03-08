#!/bin/bash

fidx='idx.txt'

echo "path,pathmsk" > $fidx
for ii in `ls -1 data/*[0-9].nii.gz | sort -n`
do
    tbn=${ii::-7}
    tmsk="${tbn}-msk.nii.gz"
    echo "${ii},${tmsk}" | tee -a $fidx
done
