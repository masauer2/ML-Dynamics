#!/bin/bash

for((i=0;i<20;i++))
do
echo "Run replica $i"
cp run_single_replica.sh run_replica_$i.sh
sed -i "11s|0|$i|" run_replica_$i.sh
sbatch run_replica_$i.sh
done  
