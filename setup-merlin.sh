#!/bin/bash
pushd solvers
url=https://github.com/radum2275/merlin/archive/008d910307a2043f9446676f28010079ea49ec15.zip
wget --quiet --show-progress --no-check-certificate $url -O merlin-src.zip
unzip merlin-src.zip
mv merlin-008d910307a2043f9446676f28010079ea49ec15 merlin-src
pushd merlin-src
patch -ub -p1 -i ../merlin.patch
make
popd
cp merlin-src/bin/merlin . -v
popd

