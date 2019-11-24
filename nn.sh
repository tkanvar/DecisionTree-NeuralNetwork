#!/bin/bash

config=$1
fileTrain=$2
fileTest=$3

py Q2.py "b" $fileTrain $fileTest "1" "1" $config
