@echo off
REM Rebuild the DoA Verification Bench from the current benchmark results.
REM Run this after deck/eval_150.py produces a new eval_150_traces.npz.
REM No server, no hosting: it writes a plain HTML file you open by double-click.
cd /d "%~dp0..\.."
echo [1/2] packing traces + table + CRLB into JSON...
".venv\Scripts\python.exe" deck\export_gui_data.py || goto :err
echo [2/2] inlining the data into the page...
python deck\gui\build_gui.py || goto :err
echo.
echo Done. Open:
echo   G:\My Drive\DOA_AI\DoA_Verification_Bench.html
echo   %CD%\deck\gui\doa_verify_inline.html
exit /b 0
:err
echo.
echo FAILED - see the message above.
exit /b 1
