#!/bin/bash
pushd solvers
wget --quiet --show-progress https://ipg.idsia.ch/upfiles/blip-publish.zip --no-check-certificate
unzip blip-publish.zip
pushd blip-publish
patch -ub -p1 -i ../blip.patch
./compile-blip.sh
popd
cp blip-publish/blip.jar . -v
popd

