#!/bin/bashi
rm -fr a.out
nvcc -O3 -pg -std=c++11 *.cpp *.cu --gpu-architecture=compute_35 --gpu-code=sm_35 -I ./include -DIFHOP=0 -DSERT=5000 -DNODE=1000
#./a.out G
#gprof -b a.out gmon.out >report.txt
