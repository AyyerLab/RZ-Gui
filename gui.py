#!/usr/bin/env python3

import sys
import os
import time
import datetime
import argparse
try: 
    from PyQt5 import QtCore, QtWidgets, QtGui
except ImportError:
    print('PyQt5 import failed')
    import sip
    sip.setapi('QString', 2)
    from PyQt4 import QtCore, QtGui
    from PyQt4 import QtGui as QtWidgets
import pyqtgraph as pg
import numpy as np
import h5py
import pandas

class RZ_gui(QtWidgets.QWidget):
    def __init__(self, h5_fname, h5_dset, det_fname):
        super(RZ_gui, self).__init__()
        self.h5_fname = h5_fname
        self.h5_dset = h5_dset
        self.det_fname = det_fname
        self.frame_num = 0
        self.rz_flag = False
        self.update_flag = True
        self.r_avg_flag = False
        self.z_avg_flag = False
        self.subtract_bg_flag = False
        self.auto_range = True
        self.frame_changed = True
        self.assem_shape = None
        self.assem = None
        self.angles = None
        self.phi = 0.
        self.beta = 0.
        self.detd = 85. / 0.11
        
        self.get_geom()
        
        self.init_UI()

    def init_UI(self):
        # Overall layout
        window = QtWidgets.QVBoxLayout()

        # RZ merge ImageView
        self.imview = pg.ImageView(self, view=pg.PlotItem())
        self.imview.scene.setClickRadius(1)
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()
        window.addWidget(self.imview, stretch=1)
        self.imview.setLevels(0,400)

        line = pg.InfiniteLine(angle=0, movable=True, pen='g')
        self.imview.addItem(line)
        line = pg.InfiniteLine(angle=90, movable=True, pen='g')
        self.imview.addItem(line)
        line = pg.InfiniteLine(angle=90, movable=True, pen='g')
        self.imview.addItem(line)

        # Options frame
        vbox = QtWidgets.QVBoxLayout()
        window.addLayout(vbox)

        # -- HDF5 file and data set
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Filename:', self)
        hbox.addWidget(label)
        entry = QtWidgets.QLineEdit(self.h5_fname, self)
        entry.textChanged.connect(self.fname_changed)
        hbox.addWidget(entry)
        label = QtWidgets.QLabel('H5 Dataset:', self)
        hbox.addWidget(label)
        entry = QtWidgets.QLineEdit(self.h5_dset, self)
        entry.textChanged.connect(self.dset_changed)
        hbox.addWidget(entry)
        label = QtWidgets.QLabel('Num:', self)
        hbox.addWidget(label)
        entry = QtWidgets.QLineEdit('0', self)
        entry.textChanged.connect(self.frame_num_changed)
        hbox.addWidget(entry)
        hbox.addStretch(1)

        # -- Sliders
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtWidgets.QLabel('Phi', self)
        hbox.addWidget(label)
        self.phi_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.phi_slider.valueChanged.connect(self.phi_changed)
        self.phi_slider.sliderReleased.connect(self.replot)
        self.phi_slider.setRange(0,720)
        hbox.addWidget(self.phi_slider)
        self.phi_val = QtWidgets.QLabel('%4.1f'%(self.phi_slider.value()/2.), self)
        hbox.addWidget(self.phi_val)
        label = QtWidgets.QLabel('    Beta', self)
        hbox.addWidget(label)
        self.beta_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.beta_slider.valueChanged.connect(self.beta_changed)
        self.beta_slider.sliderReleased.connect(self.replot)
        self.beta_slider.setRange(0,180)
        hbox.addWidget(self.beta_slider)
        self.beta_val = QtWidgets.QLabel('%4.1f'%(self.beta_slider.value()/2.), self)
        hbox.addWidget(self.beta_val)

        # -- Save and Quit buttons
        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        self.rz_button = QtWidgets.QCheckBox('RZ Embed', self)
        self.rz_button.stateChanged.connect(self.rz_flag_changed)
        hbox.addWidget(self.rz_button)
        self.update_button = QtWidgets.QCheckBox('Update Plot', self)
        self.update_button.stateChanged.connect(self.update_flag_changed)
        self.update_button.setChecked(True)
        hbox.addWidget(self.update_button)
        self.r_avg_button = QtWidgets.QCheckBox('Meridional average', self)
        self.r_avg_button.stateChanged.connect(self.r_avg_flag_changed)
        self.r_avg_button.setChecked(False)
        hbox.addWidget(self.r_avg_button)
        self.z_avg_button = QtWidgets.QCheckBox('Equatorial average', self)
        self.z_avg_button.stateChanged.connect(self.z_avg_flag_changed)
        self.z_avg_button.setChecked(False)
        hbox.addWidget(self.z_avg_button)
        self.subtract_bg_button = QtWidgets.QCheckBox('Subtract BG', self)
        self.subtract_bg_button.stateChanged.connect(self.subtract_bg_flag_changed)
        self.subtract_bg_button.setChecked(False)
        hbox.addWidget(self.subtract_bg_button)
        hbox.addStretch(1)
        button = QtWidgets.QPushButton('Save Angles', self)
        button.clicked.connect(self.save_angles)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Save', self)
        button.clicked.connect(self.save_image)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Save All', self)
        button.clicked.connect(self.save_all)
        hbox.addWidget(button)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        hbox.addWidget(button)

        self.replot()
        self.setLayout(window)
        self.setGeometry(100,100,1000,800)
        self.setWindowTitle('Manual RZ Embedding GUI')
        self.show()

    def get_image(self):
        with h5py.File(self.h5_fname, 'r') as f:
            img = f[self.h5_dset][self.frame_num].flatten()
            self.num_frames = f[self.h5_dset].shape[0]
            if self.angles is None:
                fname = os.path.splitext(self.h5_fname)[0]+'_angles.dat'
                if os.path.isfile(fname):
                    print('Reading angles from', fname)
                    self.angles = np.loadtxt(fname)
                else:
                    self.angles = np.zeros((f[self.h5_dset].shape[0], 3))
                    self.angles[:,0] = np.arange(self.angles.shape[0])
         
        self.phi_slider.setValue(self.angles[self.frame_num, 1]*2.)
        self.beta_slider.setValue(self.angles[self.frame_num, 2]*2.)
        
        return img

    def get_geom(self):
        #det = pandas.read_csv('det_lm27.dat', skiprows=1, delimiter='\t', header=None).as_matrix()
        det = pandas.read_csv(self.det_fname, skiprows=1, delim_whitespace=True, header=None).as_matrix()
        self.cx = det[:,0]
        self.cy = det[:,1]
        self.pol = det[:,2]
        self.mask = det[:,3].astype('u1')
        self.size = 2*int(np.ceil(np.sqrt(self.cx*self.cx + self.cy*self.cy).max())) + 1
        #self.ewald_rad = 200.
        self.ewald_rad = 500.
        
        norm = np.sqrt(self.cx**2 + self.cy**2 + self.detd**2)
        self.qx = self.ewald_rad * self.cx / norm
        self.qy = self.ewald_rad * self.cy / norm
        self.qz = self.ewald_rad * (self.detd / norm - 1.)

        self.dx = self.dy = self.size//2
        size = int(2*self.ewald_rad + 1)
        x, y = np.indices((size,size))
        x -= size//2; y -= size//2 
        self.intrad = np.sqrt(x*x + y*y).astype('i4')
        self.radpix = np.ones_like(self.intrad, dtype=np.bool)
        '''
        # ---- Amyloid for ewald_rad = 200. ----
        self.radpix[np.absolute(y)>40] = False
        self.radpix[np.absolute(y)<18] = False
        self.radpix[(self.intrad<=18) & (np.absolute(y)>8)] = True
        # --------------------------------------
        '''
        # ---- Amyloid for ewald_rad = 500. ----
        self.radpix[np.absolute(y)>120] = False
        self.radpix[np.absolute(y)<40] = False
        self.radpix[(self.intrad<=40) & (np.absolute(y)>20)] = True
        # --------------------------------------
        self.radcounts = np.zeros((self.intrad.max()+1))

    def replot(self):
        if not self.update_flag:
            return

        if self.frame_changed:
            self.image = self.get_image()/self.pol
            self.frame_changed = False

        if self.rz_flag:
            qx0 = np.cos(self.phi)*self.qx - np.sin(self.phi)*self.qy
            qy0 = np.cos(self.phi)*self.qy + np.sin(self.phi)*self.qx
            qz0 = self.qz
            qx1 = qx0
            qy1 = np.cos(self.beta)*qy0 - np.sin(self.beta)*qz0
            qz1 = np.cos(self.beta)*qz0 + np.sin(self.beta)*qy0
            R = np.round(np.sqrt(qx1*qx1 + qz1*qz1)).astype('i4')[self.mask<2]
            Z = np.round(qy1).astype('i4')[self.mask<2]

            if not self.r_avg_flag:
                R[qx1[self.mask<2]<0.] = -R[qx1[self.mask<2]<0.]
            size = int(2*self.ewald_rad+1) ; center = size//2
            self.rz_embed = np.zeros((size,size))
            self.weights = np.zeros_like(self.rz_embed)
            np.add.at(self.weights, [R+center, Z+center], 1)
            np.add.at(self.rz_embed, [R+center, Z+center], self.image[self.mask<2])
            if self.r_avg_flag:
                self.weights = self.weights + self.weights[::-1]
                self.rz_embed = self.rz_embed + self.rz_embed[::-1]
            if self.z_avg_flag:
                self.weights = self.weights + self.weights[:,::-1]
                self.rz_embed = self.rz_embed + self.rz_embed[:,::-1]
            self.rz_embed[self.weights>0] /= self.weights[self.weights>0]

            if self.subtract_bg_flag:
                self.radcounts.fill(0.)
                radavg = np.zeros_like(self.radcounts)
                np.add.at(self.radcounts, self.intrad[(self.weights>0) & (self.radpix)], 1)
                np.add.at(radavg, self.intrad[(self.weights>0) & (self.radpix)], self.rz_embed[(self.weights>0) & (self.radpix)])
                radavg[self.radcounts>0] /= self.radcounts[self.radcounts>0]
                self.rz_embed[self.weights>0] = self.rz_embed[self.weights>0] - radavg[self.intrad[self.weights>0]]
            self.imview.setImage(self.rz_embed, autoLevels=False, autoRange=self.auto_range, autoHistogramRange=False)
            self.auto_range = False
        else:
            rx = np.cos(self.phi)*self.cx - np.sin(self.phi)*self.cy
            ry = np.cos(self.phi)*self.cy + np.sin(self.phi)*self.cx
            rx = np.round(rx).astype('i4')[self.mask<2]
            ry = np.round(ry).astype('i4')[self.mask<2]
            self.rot_image = np.zeros((self.size, self.size))
            self.weights = np.zeros_like(self.rot_image)
            np.add.at(self.weights, [rx+self.dx, ry+self.dy], 1)
            np.add.at(self.rot_image, [rx+self.dx, ry+self.dy], self.image[self.mask<2])
            self.rot_image[self.weights>0] /= self.weights[self.weights>0]
            self.imview.setImage(self.rot_image, autoLevels=False, autoRange=self.auto_range, autoHistogramRange=False)
            self.auto_range = False

    def phi_changed(self, value=None):
        self.phi = np.pi * value / 180. / 2.
        self.phi_val.setText('%4.1f'%(value/2.))
        self.angles[self.frame_num,1] = value/2.

    def beta_changed(self, value=None):
        self.beta = np.pi * value / 180. / 2.
        self.beta_val.setText('%4.1f'%(value/2.))
        self.angles[self.frame_num,2] = value/2.

    def rz_flag_changed(self, event=None):
        self.rz_flag = self.rz_button.isChecked()
        self.auto_range = True
        self.replot()

    def update_flag_changed(self, event=None):
        self.update_flag = self.update_button.isChecked()
        self.replot()

    def r_avg_flag_changed(self, event=None):
        self.r_avg_flag = self.r_avg_button.isChecked()
        self.replot()

    def z_avg_flag_changed(self, event=None):
        self.z_avg_flag = self.z_avg_button.isChecked()
        self.replot()

    def subtract_bg_flag_changed(self, event=None):
        self.subtract_bg_flag = self.subtract_bg_button.isChecked()
        self.replot()

    def fname_changed(self, text=None):
        self.h5_fname = str(text)
        self.frame_changed = True

    def dset_changed(self, text=None):
        self.h5_dset = text
        self.frame_changed = True

    def frame_num_changed(self, text=None):
        try:
            self.frame_num = int(text)
            self.frame_changed = True
        except ValueError:
            pass

    def save_image(self):
        if self.rz_flag:
            fname = os.path.splitext(self.h5_fname)[0]+'_'+'%.3d'%self.frame_num+'_rz.npy'
            print('Saving to', fname)
            np.save(fname, self.rz_embed)
            fname = os.path.splitext(self.h5_fname)[0]+'_'+'%.3d'%self.frame_num+'_rzw.npy'
            np.save(fname, self.weights)
        else:
            fname = os.path.splitext(self.h5_fname)[0]+'_'+'%.3d'%self.frame_num+'.npy'
            print('Saving to', fname)
            np.save(fname, self.rot_image)

    def save_angles(self):
        fname = os.path.splitext(self.h5_fname)[0]+'_angles.dat'
        print('Saving angles to', fname)
        np.savetxt(fname, self.angles, fmt='%.4d %5.1f %5.1f', header='Num    Phi  Beta', comments='')

    def save_all(self):
        for i in range(self.num_frames):
            self.frame_num = i
            self.frame_changed = True
            self.replot()
            self.save_image()

    def keyPressEvent(self, event=None):
        if event.key() == QtCore.Qt.Key_Return:
            self.replot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manual alignment and RZ embedding GUI')
    parser.add_argument('h5_fname', help='Path to HDF5 file containing the frames to be aligned')
    parser.add_argument('-d', '--det_fname', help='Path to detector file (default: det_cxim2716.dat)', default='det_cxim2716.dat')
    parser.add_argument('-D', '--dset_name', help='HDF5 dataset name containing unassembled frames (default: data/calib)', default='data/calib')
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    gui = RZ_gui(args.h5_fname, args.dset_name, args.det_fname)
    sys.exit(app.exec_())
