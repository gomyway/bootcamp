Merge the following two into one package:

[cudamat](http://code.google.com/p/cudamat/) library by Vlad Mnih and [cuda-convnet](http://code.google.com/p/cuda-convnet/) library by
Alex Krizhevsky.

To build on windows 8.1:
 nmake -f Makefile_x86.win
 
Details:

1. download and install cuda
2. search and run "developer command prompt" on win8
3.  nmake -f Makefile_x86.win
4. copy cudamat (the one contains dlls) to C:\Python27\Lib\site-packages
5. compile gnumpy
wget http://www.cs.toronto.edu/~tijmen/gnumpy.py to C:\Python27\Lib\site-packages and compile
>>> import py_compile
>>> py_compile.compile('gnumpy.py')