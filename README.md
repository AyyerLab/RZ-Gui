# RZ Embedding GUI
This GUI can be used to embed diffraction frames into RZ space. The rotation
and tilt angles are set using sliders.

## Dependencies
The program is qritten in Python 2. The following modules are necessary:

 * python 2.7+
 * numpy
 * h5py
 * pyqt4.6+
 * pandas (to read detector file)
 * pyqtgraph

## Data requirements
The diffraction pattern is assumed to be an unassembled HDF5 file. The geometry
file is currently in Dragonfly format [link](https://github.com/duaneloh/Dragonfly/wiki/Data-stream-simulator#make_detector).
More formats will be supported in the future.
