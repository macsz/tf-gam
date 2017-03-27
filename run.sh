#!/usr/bin/env bash

# $1 - path
# $2 - proc

function log {
    echo "[`date`] $1" | tee -a log.txt
}

function get_face_coords_static {
    VAR=$1
    DIR=${VAR%/*}
    C=`echo "${VAR##*/}" | cut -d'.' -f1`
    cat "${DIR}/$C-coord.txt"
}
#####
PROC="cpu"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_NAME=`basename $1 | cut -d'.' -f1`
OUTPUT_DIR=$DIR/output/$BASE_NAME/`date '+%F-%H-%M'`
FACE_COORDS_STATIC=`get_face_coords_static $1`

# JAVA #
JAVA_BIN="/usr/lib/jvm/java-8-oracle/bin/java"
JAVA_MVN_BIN="/home/macsz/netbeans-8.2/java/maven/bin/mvn"
#####

if [ -z "$1" ]
then
    log "Provide path to lepton file."
    exit
fi

if [ -n "$2" ]
then
    PROC=$2
fi
log "Using $PROC as a processor"

#####
mkdir -p frames
mkdir -p output_face
mkdir -p output_nose
mkdir -p input_nose
mkdir -p $OUTPUT_DIR
#####

pushd .
cd /home/macsz/Projects/tf-gam/desktop; JAVA_HOME=/usr/lib/jvm/java-8-oracle
/home/macsz/netbeans-8.2/java/maven/bin/mvn "-Dexec.args=-classpath %classpath pg.eti.biomed.leptonreader.FileReader $DIR/$1" -Dexec.executable=/usr/lib/jvm/java-8-oracle/bin/java org.codehaus.mojo:exec-maven-plugin:1.2.1:exec
popd

START=$(date +%s.%N)
python class/classify.py \
    --face-coords-static $FACE_COORDS_STATIC | tee -a log.txt
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
log "Python code took $DIFF seconds to execute"

python class/sum_time.py log.txt | tee sum_time.txt

ffmpeg -f image2 -r 12 -i output_face/frame_%5d.jpg -vcodec mpeg4 -y face-`basename $1``date '+%F-%H-%M'`.mp4 2> /dev/null
ffmpeg -f image2 -r 12 -i output_nose/frame_%5d.jpg -vcodec mpeg4 -y nose-`basename $1``date '+%F-%H-%M'`.mp4 2> /dev/null

log "Finished: `date`"
log "$OUTPUT_DIR"

#####
mv frames $OUTPUT_DIR/
mv output_face $OUTPUT_DIR/
mv output_nose $OUTPUT_DIR/
mv input_nose $OUTPUT_DIR/
mv *.mp4 $OUTPUT_DIR/
mv log.txt $OUTPUT_DIR/
mv sum_time.txt $OUTPUT_DIR/
mv grid.jpg $OUTPUT_DIR/$BASE_NAME.jpg
#####

# TODO add other file managers; for OSX etc...
#dolphin $OUTPUT_DIR &
