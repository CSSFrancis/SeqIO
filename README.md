# SeqIO

This package is a lightweight package for loading files in the .seq file format used by DE Streampix into hyperspy.  It includes the ability to read files as 3D 
signals (time, x, y), 4d Signals (x,y,kx,ky) and 5d signals (x,y,t,kx,ky). 

It reads the metadata given to Streampix but additional effort to caputure micropsope set up etc will be an additional focus as time goes on.  Working with the scan
software will also be useful.

To download the package you can either clone the repo or the package is hosted on PyPi: so running `pip install seqio` will install the package.

From there 

```python
import SeqIO

SeqIO.load(filename='test.seq', lazy=False, chunks=None, nav_shape=None) # 3D signal not lazy
SeqIO.load(filename='test.seq', lazy=False, chunks=None, nav_shape=[4,5]) # 4D signal not lazy
SeqIO.load(filename='test.seq', lazy=False, chunks=None, nav_shape=[4,5,5]) # 5D signal not lazy
SeqIO.load(filename='test.seq', lazy=True, chunks=10, nav_shape=[4,5]) # 4D signal lazy with 10 chunks 
```


(Version 0.05 Update May 3, 2021) -- Support for loading of bottom/top images for the DE Celeritas Camera.  There are some quirks to loading this kind of data so any bug reports are appricated.
