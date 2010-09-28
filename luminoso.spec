# -*- mode: python -*-
version=1.3
a = Analysis([os.path.join(HOMEPATH,'support/_mountzlib.py'),
os.path.join(HOMEPATH,'support/useUnicode.py'), '/Users/rspeer/code/luminoso/luminoso/app.py'],
             pathex=['/Users/rspeer/py/luminoso/src/pyinstaller', 
                     '/Users/rspeer/code/luminoso/luminoso/lib/standalone_nlp',
                     '/Users/rspeer/code/luminoso/luminoso/lib',
                     '/Users/rspeer/code/luminoso',
                    ])
pyz = PYZ(a.pure)
exe = EXE( pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name=os.path.join('dist', 'luminoso'),
          debug=False,
          strip=False,
          upx=True,
          console=1 )
app = BUNDLE(exe,
    name=os.path.join('dist', 'Luminoso.app'),
    version=version)
