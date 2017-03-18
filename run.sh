#!/usr/bin/env bash

# $1 - path
# $2 - proc

PROC="cpu"

if [ -n "$2" ]
then
    PROC=$2
fi

DIRNAME=../dl_out/`basename $1`/`date '+%F-%H-%M'`

mkdir -p frames
mkdir -p output_face
mkdir -p output_nose
mkdir -p input_nose

pushd .
cd /home/macsz/Projects/deep_learning/desktop; JAVA_HOME=/usr/lib/jvm/java-8-oracle /home/macsz/netbeans-8.2/java/maven/bin/mvn "-Dexec.args=-classpath %classpath pg.eti.biomed.leptonreader.FileReader $1" -Dexec.executable=/usr/lib/jvm/java-8-oracle/bin/java org.codehaus.mojo:exec-maven-plugin:1.2.1:exec
popd

START=$(date +%s.%N)
python class/classify.py | tee time.txt
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo Python code took $DIFF seconds to execute | tee -a time.txt

ffmpeg -f image2 -r 12 -i output_face/frame_%5d.jpg -vcodec mpeg4 -y face-`basename $1``date '+%F-%H-%M'`.mp4 2> /dev/null
ffmpeg -f image2 -r 12 -i output_nose/frame_%5d.jpg -vcodec mpeg4 -y nose-`basename $1``date '+%F-%H-%M'`.mp4 2> /dev/null

mkdir -p $DIRNAME
mv frames $DIRNAME/
mv output_face $DIRNAME/
mv output_nose $DIRNAME/
mv input_nose $DIRNAME/
mv *.mp4 $DIRNAME/
mv time.txt $DIRNAME/

echo Finished: `date`
echo $DIRNAME

# TODO add other file managers; for OSX etc...
#dolphin $DIRNAME &
