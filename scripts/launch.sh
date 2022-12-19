#!/bin/sh

# This script will launch python 21-train.py in a loop (everytime it finishes)
# It will also log the output to a file

# This is the path to the python script
SCRIPT=21-train.py
LOGNUM=0
# test if the log file exists and find the next available number
while true; do
	while [ -f logs/log-$LOGNUM.txt ]
	do
		LOGNUM=$((LOGNUM+1))
	done
	LOGFILE=logs/log-$LOGNUM.txt
	echo "Starting $SCRIPT, logging to $LOGFILE"
	python $SCRIPT | tee $LOGFILE
	LOGNUM=$((LOGNUM+1))
done
