# clean output files in out/*.csv
find out -mindepth 3 -delete
rm ./out/unfiltered/*.csv
mkdir ./out/filtered/L2/corrected
mkdir ./out/filtered/L2/uncorrected
mkdir ./out/filtered/SGS/R
mkdir ./out/filtered/SGS/S
mkdir ./out/filtered/SGS/Tau