@echo off
set EXE=octree_Win32_Release.exe
if not exist %EXE% set EXE=octree.exe
set LOG=build_and_benchmark.log
set SCENES=scenes
set OCTREES=octrees
set TMP_OCT=%OCTREES%\tmp.oct

rem Build conference.

%EXE% build --log=%LOG% --in=%SCENES%\conference\conference.obj --out=%TMP_OCT% --levels=14 --contours=1 --color-error=256 --normal-error=0.0101353 --contour-error=14.0421 --max-threads=4
%EXE% optimize --log=%LOG% --in=%TMP_OCT% --out=%OCTREES%\conference_14.oct
%EXE% ambient --log=%LOG% --in=%TMP_OCT% --ao-radius=0.05 --flip-normals=1
%EXE% optimize --log=%LOG% --in=%TMP_OCT% --out=%OCTREES%\conference_14_ao.oct

rem Build default.

%EXE% build --log=%LOG% --in=%SCENES%\default\default.obj --out=%TMP_OCT% --levels=12 --contours=1 --color-error=16 --normal-error=0.01 --contour-error=15 --max-threads=4
%EXE% optimize --log=%LOG% --in=%TMP_OCT% --out=%OCTREES%\default_12.oct
%EXE% ambient --log=%LOG% --in=%TMP_OCT% --ao-radius=0.15 --flip-normals=0
%EXE% optimize --log=%LOG% --in=%TMP_OCT% --out=%OCTREES%\default_12_ao.oct

rem Build hairball.

%EXE% build --log=%LOG% --in=%SCENES%\hairball\hairball.obj --out=%TMP_OCT% --levels=10 --contours=1 --color-error=256 --normal-error=0.200165 --contour-error=13 --max-threads=2
%EXE% optimize --log=%LOG% --in=%TMP_OCT% --out=%OCTREES%\hairball_10.oct
%EXE% ambient --log=%LOG% --in=%TMP_OCT% --ao-radius=0.04 --flip-normals=0
%EXE% optimize --log=%LOG% --in=%TMP_OCT% --out=%OCTREES%\hairball_10_ao.oct

rem Benchmark.

%EXE% benchmark --log=%LOG% --in=%OCTREES%\conference_14.oct --camera="3RgEz/h:ws/0DybW02/2dumy1kttDz/8PUQF/0///X10eGyZw/szbL100" --camera=":qNmz/tIHg/0c0Hjz1/ys:Yy1H4RFz18PUQF/0///X10eGyZw/szbL100" --camera="0X2Z00Ky/R/0mJ14023nwNGy1aoB0y18PUQF/0///X10eGyZw/szbL100" --camera="XyAH00o:5M/0glVO02/uW6ox18z5Ky/8PUQF/0///X10eGyZw/szbL100" --camera="8xep00jG78/0lNob0216gczy11KBMw18PUQF/0///X10eGyZw/szbL100" --camera="bLer00Ru0200Yb8y/23seh4z1YiJNy18PUQF/0///X10eGyZw/szbL100" --camera="1IfV00YdtX/0vQAe0217nY2x1d6Q4y18PUQF/0///X10eGyZw/szbL100" --camera="QOdl/0Wts5/08U1c02/EZPZx114MFz/8PUQF/0///X10eGyZw/szbL100" --camera="rkcd00f6Q/006H7T024xOFxy/6oo8z18PUQF/0///X10eGyZw/szbL100" --camera="CB8X008VaE/05BjR023lA5:x1QruOz/8PUQF/0///X10eGyZw/szbL100"
%EXE% benchmark --log=%LOG% --in=%OCTREES%\default_12.oct --camera="5O12/0p7Ayz/dlDNz13lQngy17ssby/8UJJJz////X108Qx7w/6//m100" --camera="msqAz1TJO5z/:XjRz13fLOcy1jVJ8z/86Trey////X108Qx7w/6//m100" --camera="TKOvy/fTr5z/SR7vy/34J1Fy/:20Dz186Trey////X108Qx7w/6//m100" --camera="OrNLz//zbbv/7GgEz1198hzw/KipBz186Trey////X108Qx7w/6//m100" --camera="ZTrRz/yzXTw/rlSEz11SQD0y/5lGJz18n4R1y////X108Qx7w/6//m100" --camera="sLQwz14u/Kz/UGtEz1/1:X0z1Iszny/87Trey////X108Qx7w/6//m100" --camera="eZBUy1Wlu6/0aKSfy14NiYhx/3fQ2u/87Trey////X108Qx7w/6//m100" --camera="iR:9z1so2cz/DobXz11VIyGy/An7Lz187Trey////X108Qx7w/6//m100" --camera="qTHLz/ZdYdz/jaGMz/4/ii1z1VoFCz187Trey////X108Qx7w/6//m100" --camera="duW3x/Npmrw/8StDz/5iCUNx1xlbRx/87Trey////X108Qx7w/6//m100"
%EXE% benchmark --log=%LOG% --in=%OCTREES%\hairball_10.oct --camera="QU4Wy/7jF6/0Gx7u/05MV5dv1ydgey18ZMj3/0///X10W6nZv/AcLL000" --camera="X64Ow1NFB7/0uicw/21d3GDx1iAicy18ZMj3/0///X10W6nZv/AcLL000" --camera="Zh:M/27jF6/0:Y3p/21TGivy/zPZdy18ZMj3/0///X10W6nZv/AcLL000" --camera="ODwo/27jF6/0YUuJ/2/nOegy1ZLV3z/8ZMj3/0///X10W6nZv/AcLL000" --camera="dqht/27jF6/0qxo//0/Kjpdy17InUy18ZMj3/0///X10W6nZv/AcLL000" --camera="T38V/27jF6/08nSk/05z3LBz/Fa4my18ZMj3/0///X10W6nZv/AcLL000" --camera="eOiV/07jF6/034ai/05DXy7z1X7Hny18ZMj3/0///X10W6nZv/AcLL000" --camera="SXIp/0B:57/0Ob22/03HQ9ky15uPay18ZMj3/0///X10W6nZv/AcLL000" --camera="b6Qn/0B:57/05Lvxz13bNaiy1B7Kdy/8ZMj3/0///X10W6nZv/AcLL000" --camera="HuPY/0ZDs5/0LjzW/23Zs5vy1YQWRz/8ZMj3/0///X10W6nZv/AcLL000"
