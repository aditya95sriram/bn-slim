#!/bin/bash
pushd solvers
wget --quiet --show-progress https://maxsat-evaluations.github.io/2019/mse19-solver-src/complete/UWrMaxSat-1.0.zip
unzip UWrMaxSat-1.0.zip
pushd UWrMaxSat-1.0/code
patch -ub uwrmaxsat/MsSolver.cc -i ../../uwrmaxsat.patch
chmod 777 starexec_build
./starexec_build
popd
cp UWrMaxSat-1.0/bin/uwrmaxsat . -v
popd

