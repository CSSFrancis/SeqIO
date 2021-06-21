# SeqIO

This package is a lightweight package for loading files in the .seq file format used by DE Streampix into hyperspy.  It includes the ability to read files as 3D 
signals (time, x, y), 4d Signals (x,y,kx,ky) and 5d signals (x,y,t,kx,ky). 

It reads the metadata given to Streampix but additional effort to caputure micropsope set up etc will be an additional focus as time goes on.  Working with the scan
software will also be useful.

To download the package you can either clone the repo or the package is hosted on PyPi: so running `pip install seq-io` will install the package.

From there 

```python
import SeqIO

SeqIO.load(filename='test.seq', lazy=False, chunks=None, nav_shape=None) # 3D signal not lazy
SeqIO.load(filename='test.seq', lazy=False, chunks=None, nav_shape=[4,5]) # 4D signal not lazy
SeqIO.load(filename='test.seq', lazy=False, chunks=None, nav_shape=[4,5,5]) # 5D signal not lazy
SeqIO.load(filename='test.seq', lazy=True, chunks=10, nav_shape=[4,5]) # 4D signal lazy with 10 chunks 
```


(Version 0.05 Update May 3, 2021) -- Support for loading of bottom/top images for the DE Celeritas Camera.  There are some quirks to loading this kind of data so any bug reports are appricated.

(Version 0.07 Update June 21, 2021) -- Better for dealing with lazy chunking than pervious versions. (This part is still under some development especially with very large datasets)

In addition there is a command line interface for SeqIO as well.  If you want to use the command line interface 
the easiest way to do this is to `$git clone https://github.com/CSSFrancis/SeqIO.git` The repository and then
set up a bash file which looks like this... 


```bash

#!/bin/sh
#SBATCH --time=0-04:30:00		# run time in days-hh:mm:ss
#SBATCH --nodes=1			# require 1 nodes
#SBATCH --ntasks-per-node=20            # (by default, "ntasks"="cpus")
#SBATCH --gres=gpu:0                    # Number of GPUS
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#Make sure to change the above two lines to reflect your appropriate
# file locations for standard error and output

#Now list your executable command (or a string of them).
# Example for non-SLURM-compiled code:
module load cuda/10.2

python3 process.py -ns 200 120 120 -d '/srv/home/csfrancis/4d_STEM/2021/2021-05-24CSF/SS10/SS102pt51NoCDS' -t 6 -c 
```
