#!/usr/bin/env bash

if ! command -v convert &> /dev/null
then
    echo "convert command could not be found, please install imagemagick."
    exit
fi


cd 'data/DeepFashion/'
mkdir img_resized
for foldername in `ls img`; do
  for file in `ls img/$foldername/*.jpg`; do
    infile=$file
    outfolder=img_resized/$foldername
    if [ ! -d $outfolder ]; then
      mkdir $outfolder
    fi
    outfile=$outfolder/`basename $file`
    if [ ! -f $outfile ]; then
      (convert $infile -fuzz 5% -trim -resize 256x256 -gravity center -extent 256x256 $outfile) &
    fi
  done
  wait
  echo "Resized $foldername"
done