#!/bin/bash

tdir="$PWD/tmp-out"

find $PWD -name 'raw' | while read rdir
do
    echo ":: [${rdir}]"
    mkdir -p "${tdir}"
    numfn=`find ${rdir} -name '*.dcm' | wc -l`
    ttype=`find ${rdir} -name '*.dcm' | sort -n | head -n 1`
    ttype=`basename "${ttype}" | cut -d\-  -f2`
    if [[ "$numfn" < "1" ]]; then
	echo "ok : $rdir"
    fi
    ##ttype=''
    if [ -n "${ttype}" ]; then
	dser=`dirname ${rdir}`
	idxs=`basename ${dser}`
	echo -e "\t--> [${ttype}] : ${idxs}"
	fout="${dser}-${ttype}.nii.gz"
	echo -e "\t--> [${fout}]"
	pushd "${rdir}/.."
	echo "------"
	pwd
	echo "------"
	dcm2nii -r n -o ./ ./raw
	finp=`find ./ -name '*.nii.gz' | sort -n | head -n 1`
	if [ -n "${finp}" ]; then
	    mv "${finp}" "${fout}" 
	fi
	popd
	rm -f ${tdir}/*
    else
	:
##	echo "**WARNING** dir is empty [${rdir}]"
    fi
#    exit 1
done
