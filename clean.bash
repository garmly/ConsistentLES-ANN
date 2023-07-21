# clean output files in out/*.csv
find out -mindepth 3 -delete
rm ./out/unfiltered/*.csv
mkdir ./out/filtered/L2/corrected
mkdir ./out/filtered/L2/uncorrected