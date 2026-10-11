# -*- mode: python ; coding: utf-8 -*-
# BossAgent 打包配置：onedir 模式，保留控制台查看日志
# 用法: pyinstaller BossAgent.spec --noconfirm
from PyInstaller.utils.hooks import collect_all

datas = [
    ('templates', 'templates'),
    ('data/city_codes.json', 'data'),
    ('data/degree_codes.json', 'data'),
    ('data/experience_codes.json', 'data'),
    ('data/job_type_codes.json', 'data'),
    ('data/money_codes.json', 'data'),
    ('data/scale_codes.json', 'data'),
]
binaries = []
hiddenimports = [
    # uvicorn 运行时按字符串动态导入，必须手动声明
    'uvicorn.logging',
    'uvicorn.loops.auto',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
    'langchain_openai',
    'openai',
]

# langchain / DrissionPage 生态含大量数据文件与子模块，整体收集最稳妥
for pkg in ('langchain_core', 'langchain_openai', 'DrissionPage'):
    tmp_datas, tmp_binaries, tmp_imports = collect_all(pkg)
    datas += tmp_datas
    binaries += tmp_binaries
    hiddenimports += tmp_imports

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BossAgent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='BossAgent',
)
