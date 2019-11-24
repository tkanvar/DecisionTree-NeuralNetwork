#!/bin/bash

fileTrain=$1
fileTest=$2
outtrain=$3 
outtest=$4

py Q2.py "a" $fileTrain $fileTest $outtrain $outtest "1"
