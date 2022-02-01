#!bash
DIR=$1
ls -lh $DIR | awk -v dir=$DIR '$5==0 {print "rm " dir "/" $NF}'