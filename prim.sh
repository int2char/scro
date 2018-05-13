#!/bin/bash
rm -fr a.out
nvcc -O3 -pg -std=c++11 *.cpp *.cu --gpu-architecture=compute_35 --gpu-code=sm_35 -I ./include -DIFHOP=0 -DSERT=20000 -DNODE=100
#./a.out G
#gprof -b a.out gmon.out >report.txt
