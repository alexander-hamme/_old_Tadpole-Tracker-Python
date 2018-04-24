#!/bin/bash

trap "exit" INT

for file in ./darknet/backup/*.weights 
do
  python evaluate_models.py $file "evaluations.txt" > /dev/null -e
done

