A# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('mediapipe')
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('cvzone')
hiddenimports += collect_submodules('tkinter')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('math')
hiddenimports += collect_submodules('time')
hiddenimports += collect_submodules('pathlib')


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\rukos\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\utils', 'utils'), ('C:\\Users\\rukos\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\mediapipe', 'mediapipe'), ('C:\\Users\\rukos\\PycharmProjects\\диплом2\\build\\assets\\frame0', 'frame0'), ('C:\\Users\\rukos\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\tkinter', 'tkinter'), ('C:\\Users\\rukos\\PycharmProjects\\диплом2\\build\\utils.py', 'utils.py'), ('C:\\Users\\rukos\\PycharmProjects\\диплом2\\build\\assets', 'assets'), (C:\Users\rukos\PycharmProjects\диплом2\build\icon.ico)],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    icon='C:\Users\rukos\PycharmProjects\диплом2\build\icon.ico',
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
