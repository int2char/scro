#!/bin/bash
rm -fr a.out *.txt
sers=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000)
for s in ${sers[@]}
do
	nvcc -O3 -pg -std=c++11 *.cpp *.cu --gpu-architecture=compute_35 --gpu-code=sm_35 -I ./include -DSERT=$s -DIFHOP=0
	CUDA_VISIBLE_DEVICES=0 ./a.out F 1>>FDATA.txt
	#CUDA_VISIBLE_DEVICES=0 ./a.out P 1>>PDATA.txt
	#CUDA_VISIBLE_DEVICES=0 ./a.out S 1>>SDATA.txt
	#CUDA_VISIBLE_DEVICES=0 ./a.out G 1>>GDATA.txt
	#rm -fr a.out
	#nvcc -O3 -pg -std=c++11 *.cpp *.cu --gpu-architecture=compute_35 --gpu-code=sm_35 -I ./include -DSERT=$s -DIFHOP=1
	#CUDA_VISIBLE_DEVICES=0 ./a.out F 1>>HFDATA.txt
        #CUDA_VISIBLE_DEVICES=0 ./a.out P 1>>HPDATA.txt
        #CUDA_VISIBLE_DEVICES=0 ./a.out S 1>>HSDATA.txt
        #CUDA_VISIBLE_DEVICES=0 ./a.out G 1>>HGDATA.txt
done
#gprof -b a.out gmon.out >report.txt
