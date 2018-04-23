# RZ Embedding GUI
This GUI can be used to embed diffraction frames into RZ space. The rotation
and tilt angles are set using sliders.

## Dependencies
The program is written in Python 3. The following modules are necessary:

 * python 3
 * numpy
 * h5py
 * PyQt4 or PyQt5
 * pandas (to read detector file)
 * pyqtgraph

## Data requirements
The diffraction pattern is assumed to be an unassembled HDF5 file. The geometry
file is currently in Dragonfly format ([description](https://github.com/duaneloh/Dragonfly/wiki/Data-stream-simulator#make_detector)).
More formats will be supported in the future on request.

## Usage
Specify the HDF5 file name and data set name containing diffraction frame(s).
This frame is plotted using the supplied geometry file in detector space. 

```
$ ./gui.py -d <detector_fname> -D <h5_dataset_name> <h5_fname>
```

The description of the options can be seen by running `./gui.py -h`.

### Aligning frames
Here is a simple step-by-step guide to aligning a diffraction pattern:

 - The 'Phi' slider should be used to align the frame in-plane such that the layer
lines are symmetric about the vertical axis (meridian). Note that the equatorial
layer line need not be straight.

 - After moving the 'Phi' slider, turn on RZ embedding and tune the 'Beta' slider
till the layer lines are horizontal. If the layer lines are curving the wrong
way on increasing beta, add or subtract 180 to phi.
The horizontal and vertical lines are guides to the eye and can be dragged.

 - Toggle equatorial and meridional averaging. In the ideal case, this should not
 change the pattern except to make it less noisy. If features from opposite sides
 do not match, you can tweak the angles (or you may have asymmetric diffraction).

