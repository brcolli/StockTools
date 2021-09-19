@echo off
set /p ui="Enter UI file name: "

python -m PyQt5.uic.pyuic styles\%ui%.ui -o src\%ui%.py -x
