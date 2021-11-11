#!/bin/bash
. /home/klklp98/speechst2/zeroth_n/zeroth/s5/cmd.sh
. /home/klklp98/speechst2/zeroth_n/zeroth/s5/path.sh

cd /home/klklp98/speechst2/zeroth_n/zeroth/s5/EXTRACT
./decode.sh $1

echo `pwd`

lat=result/lat.1.gz
model=test/models/korean/zeroth/final.mdl

lattice-align-phones $model "ark:gunzip -c $lat|" ark:1.lats
lattice-1best ark:1.lats ark:2.lats
nbest-to-linear ark:2.lats ark:1.ali
ali-to-phones --ctm-output $model ark:1.ali phones.ctm 