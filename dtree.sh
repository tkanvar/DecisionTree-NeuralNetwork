#!/bin/bash

partnum=$1
fileTrain=$2
fileTest=$3
fileValid=$4

py Q1.py $partnum $fileTrain $fileTest $fileValid
