# RZ Embedding GUI
This GUI can be used to embed diffraction frames into RZ space. The rotation
and tilt angles are set using sliders.

## Dependencies
The program is written in Python 2. The following modules are necessary:

 * python 2.7+
 * numpy
 * h5py
 * pyqt4.6+
 * pandas (to read detector file)
 * pyqtgraph

## Data requirements
The diffraction pattern is assumed to be an unassembled HDF5 file. The geometry
file is currently in Dragonfly format ([description](https://github.com/duaneloh/Dragonfly/wiki/Data-stream-simulator#make_detector)).
More formats will be supported in the future.

## Usage
Specify the HDF5 file name and data set name containing diffraction frame(s).
If there are multiple frames, one can give the frame number (default=0). This
frame is plotted using the supplied geometry file in detector space. 

The 'Phi' slider should be used to align the frame in-plane such that the layer
lines are symmetric about the vertical axis (meridian). Note that the equatorial
layer line need not be straight.

After moving the 'Phi' slider, turn on RZ embedding and tune the 'Beta' slider
till the layer lines are horizontal. If the layer lines are curving the wrong
way on increasing beta, add or subtract 180 to phi.

The horizontal line is a guide to the eye and can be dragged vertically.
