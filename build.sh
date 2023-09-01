#!/bin/zsh
echo "Building"
export CC=gcc-10
export CXX=g++-10
conda run -n hypergan python setup.py build
echo "Done"