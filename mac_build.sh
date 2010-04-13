#!/bin/sh
# usage: mac_build.sh <version>

rm -r dist/Luminoso.app
rm dist/Luminoso-$1.dmg
rm -r DMGSkeleton/Luminoso.app
python setup.py py2app \
&& macdeployqt dist/Luminoso.app \
&& cp -r icons/ dist/Luminoso.app/Contents/MacOS/icons \
&& cp -r study_skel/ dist/Luminoso.app/Contents/MacOS/study_skel \
&& cp -r dist/Luminoso.app DMGSkeleton/Luminoso.app \
&& hdiutil create dist/Luminoso-$1.dmg -volname "Luminoso $1" -fs HFS+ -srcfolder DMGSkeleton

