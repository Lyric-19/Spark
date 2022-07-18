#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 4 #number of cores
#$ -l rmem=10G #number of memery
#$ -P rse-com6012 # require a com6012-reserved node
#$ -q rse-com6012.q # specify com6012 queue
#$ -o ../Output/Q1_code_output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 10g --executor-memory 10g --master local[4] ../Code/Q1_code.py  # .. is a relative path, meaning one level up
