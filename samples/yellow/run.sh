#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python3 yellow.py --dataset='/home/public/tangdi/yellowset/' train
