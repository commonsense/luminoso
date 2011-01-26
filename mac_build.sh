#!/bin/sh
if [ -e dist/Luminoso.app ]; then
    rm -r dist/Luminoso.app
fi
if [ -e dist/luminoso ]; then
    rm -r dist/luminoso
fi
if [ -e build ]; then
    rm -r build
fi
if [ -e DMGSkeleton/Luminoso.app ]; then
    rm -r DMGSkeleton/Luminoso.app
fi
if [ -e dist/Luminoso-$1.dmg ]; then
    rm -r dist/Luminoso-$1.dmg
fi
python pyinstaller/Build.py luminoso.spec \
&& cp -r /Library/Frameworks/QtGui.framework/Versions/4/Resources/qt_menu.nib dist/Luminoso.app/Contents/Resources/ \
&& cp -a dist/luminoso/* dist/Luminoso.app/Contents/MacOS/ \
&& cp -a Info.plist dist/Luminoso.app/Contents/ \
&& cp icons/luminoso.icns dist/Luminoso.app/Contents/Resources/ \
&& cp -r icons dist/Luminoso.app/Contents/MacOS/ \
&& cp -r dist/Luminoso.app DMGSkeleton/Luminoso.app \
&& hdiutil create dist/Luminoso-$1.dmg -volname "Luminoso $1" -fs HFS+ -srcfolder DMGSkeleton \
&& rm warnluminoso*\
&& scp dist/Luminoso-$1.dmg anemone.media.mit.edu:/var/www/conceptnet/dist/Luminoso/
