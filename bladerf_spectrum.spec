# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],  # Ваш файл называется main.py или bladerf_spectrum.py?
    pathex=[],
    binaries=[
    ('C:/Users/vova/PycharmProjects/Spectrum_analyzer_for_bladerf_2_0_windows/bladerf', '.')
    ],
    datas=[
        ('bladerf2_0.ico', '.'),
    ],
    hiddenimports=[
        'numpy',
        'numpy.core',
        'numpy.core._multiarray_umath',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'bladerf',
        'bladerf._bladerf',
        'cffi',              # ДОБАВЛЕНО
        '_cffi_backend',     # ДОБАВЛЕНО
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BladeRF_Spectrum_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Оставьте True для отладки, потом можно поменять на False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='bladerf2_0.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BladeRF_Spectrum_Analyzer',
)