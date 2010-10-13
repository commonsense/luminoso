# -*- mode: python -*-
version = "1.3.2"
a = Analysis([os.path.join(HOMEPATH,'support/_mountzlib.py'),
os.path.join(HOMEPATH,'support/useUnicode.py'), 'luminoso/app.py'],
             pathex=['pyinstaller', 
                     'luminoso/lib/standalone_nlp',
                     'luminoso/lib',
                     '.',
                    ])
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=1,
          name=os.path.join('build/pyi.darwin/luminoso', 'luminoso'),
          debug=False,
          strip=False,
          upx=True,
          console=1 )
coll = COLLECT( exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name=os.path.join('dist', 'luminoso'))
app = BUNDLE(exe,
    name=os.path.join('dist', 'Luminoso.app'),
    version=version)
