#!/bin/bash
# The benchmarks of all toolkits 
python benchmark.py -config ./configs/mxnet_bm2cpu1.config -post True
python benchmark.py -config ./configs/mxnet_bm2cpu2.config -post True
python benchmark.py -config ./configs/mxnet_bm2cpu4.config -post True
python benchmark.py -config ./configs/mxnet_bm2cpu8.config -post True
python benchmark.py -config ./configs/mxnet_bm2cpu16.config -post True
python benchmark.py -config ./configs/mxnet_bm2cpu32.config -post True
