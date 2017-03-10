#!/usr/bin/env bash

DIRNAME=out/`basename $1`/`date '+%F-%H-%M'`

mkdir -p frames
mkdir -p output_face
mkdir -p output_nose
mkdir -p input_nose

pushd .
cd /home/macsz/Projects/deep_learning/desktop; JAVA_HOME=/usr/lib/jvm/java-8-oracle /home/macsz/netbeans-8.2/java/maven/bin/mvn "-Dexec.args=-classpath %classpath pg.eti.biomed.leptonreader.FileReader $1" -Dexec.executable=/usr/lib/jvm/java-8-oracle/bin/java org.codehaus.mojo:exec-maven-plugin:1.2.1:exec
popd
python class/classify.py

ffmpeg -f image2 -r 12 -i output_face/frame_%d.jpg -vcodec mpeg4 -y face-`basename $1``date '+%F-%H-%M'`.mp4
ffmpeg -f image2 -r 12 -i output_nose/frame_%d.jpg -vcodec mpeg4 -y nose-`basename $1``date '+%F-%H-%M'`.mp4

mkdir -p $DIRNAME
mv frames $DIRNAME/
mv output_face $DIRNAME/
mv output_nose $DIRNAME/
mv input_nose $DIRNAME/

echo Finished: `date`
