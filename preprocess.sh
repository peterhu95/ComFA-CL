#!/bin/bash
mkdir log
mkdir torch_saved
mkdir data

unzip data_compressed/FB15k-237.zip -d data
unzip data_compressed/WN18RR.zip -d data
unzip data_compressed/YAGO3-10.zip -d data
unzip data_compressed/NELL-995.zip -d data