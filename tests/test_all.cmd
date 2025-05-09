
ECHO OFF

cd ..\src
echo application source directory is: %cd%
set test_dir=%cd%\..\tests
echo test dir is %test_dir%

echo running tests...

python -m unittest discover %test_dir%