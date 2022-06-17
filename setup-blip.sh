#!/bin/bash
pushd solvers

# obtain blip source code
wget --quiet --show-progress https://ipg.idsia.ch/upfiles/blip-publish.zip --no-check-certificate
unzip blip-publish.zip
pushd blip-publish

# patch and compile first for cwidth version
patch -ub -p1 -i ../blip.patch
./compile-blip.sh
cp blip.jar ../blip.jar -v

# undo patch
patch -u -p1 -i ../blip.patch --reverse

# patch and compile for constrained version
patch -ub -p1 -i ../blip-con.patch
./compile-blip.sh
cp blip.jar ../blip-con.jar -v

# reset cwd
popd
popd

