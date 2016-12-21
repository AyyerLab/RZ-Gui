#!/usr/bin/env python

import sys
import os
import time
import datetime
from PyQt4 import QtGui
from PyQt4 import QtCore
import pyqtgraph as pg
import numpy as np
import h5py
import pandas

class RZ_gui(QtGui.QWidget):
    def __init__(self, h5_fname, h5_dset):
        super(RZ_gui, self).__init__()
        self.h5_fname = h5_fname
        self.h5_dset = h5_dset
        self.frame_num = 0
        self.rz_flag = False
        self.update_flag = True
        self.r_avg_flag = False
        self.auto_range = True
        self.frame_changed = True
        self.assem_shape = None
        self.assem = None
        self.phi = 0.
        self.beta = 0.
        self.detd = 90. / 0.11
        
        self.get_geom()
        
        self.init_UI()

    def init_UI(self):
        # Overall layout
        window = QtGui.QVBoxLayout()

        # RZ merge ImageView
        self.imview = pg.ImageView(self)
        self.imview.scene.setClickRadius(1)
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()
        window.addWidget(self.imview, stretch=1)
        self.replot()
        self.imview.setLevels(0,400)

        line = pg.InfiniteLine(angle=0, movable=True, pen='g')
        self.imview.addItem(line)

        # Options frame
        vbox = QtGui.QVBoxLayout()
        window.addLayout(vbox)

        # -- HDF5 file and data set
        hbox = QtGui.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtGui.QLabel('Filename:', self)
        hbox.addWidget(label)
        entry = QtGui.QLineEdit(self.h5_fname, self)
        entry.textChanged.connect(self.fname_changed)
        hbox.addWidget(entry)
        label = QtGui.QLabel('H5 Dataset:', self)
        hbox.addWidget(label)
        entry = QtGui.QLineEdit(self.h5_dset, self)
        entry.textChanged.connect(self.dset_changed)
        hbox.addWidget(entry)
        label = QtGui.QLabel('Num:', self)
        hbox.addWidget(label)
        entry = QtGui.QLineEdit('0', self)
        entry.textChanged.connect(self.frame_num_changed)
        hbox.addWidget(entry)
        hbox.addStretch(1)

        # -- Sliders
        hbox = QtGui.QHBoxLayout()
        vbox.addLayout(hbox)
        label = QtGui.QLabel('Phi', self)
        hbox.addWidget(label)
        self.phi_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.phi_slider.valueChanged.connect(self.phi_changed)
        self.phi_slider.sliderReleased.connect(self.replot)
        self.phi_slider.setRange(0,360)
        hbox.addWidget(self.phi_slider)
        self.phi_val = QtGui.QLabel('%.3d'%self.phi_slider.value(), self)
        hbox.addWidget(self.phi_val)
        label = QtGui.QLabel('    Beta', self)
        hbox.addWidget(label)
        self.beta_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.beta_slider.valueChanged.connect(self.beta_changed)
        self.beta_slider.sliderReleased.connect(self.replot)
        self.beta_slider.setRange(0,180)
        hbox.addWidget(self.beta_slider)
        self.beta_val = QtGui.QLabel('%4.1f'%self.beta_slider.value(), self)
        hbox.addWidget(self.beta_val)

        # -- Save and Quit buttons
        hbox = QtGui.QHBoxLayout()
        vbox.addLayout(hbox)
        self.rz_button = QtGui.QCheckBox('RZ Embed', self)
        self.rz_button.stateChanged.connect(self.rz_flag_changed)
        hbox.addWidget(self.rz_button)
        self.update_button = QtGui.QCheckBox('Update Plot', self)
        self.update_button.stateChanged.connect(self.update_flag_changed)
        self.update_button.setChecked(True)
        hbox.addWidget(self.update_button)
        self.r_avg_button = QtGui.QCheckBox('Meridian average', self)
        self.r_avg_button.stateChanged.connect(self.r_avg_flag_changed)
        self.r_avg_button.setChecked(False)
        hbox.addWidget(self.r_avg_button)
        hbox.addStretch(1)
        button = QtGui.QPushButton('Save', self)
        button.clicked.connect(self.save_image)
        hbox.addWidget(button)
        button = QtGui.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        hbox.addWidget(button)

        self.setLayout(window)
        self.setGeometry(100,100,1000,800)
        self.setWindowTitle('Manual RZ Embedding GUI')
        self.show()

    def get_image(self):
        with h5py.File(self.h5_fname, 'r') as f:
            img = f[self.h5_dset][self.frame_num].flatten()
        return img

    def get_geom(self):
        det = pandas.read_csv('det_lm27.dat', skiprows=1, delimiter='\t', header=None).as_matrix()
        self.cx = det[:,0]
        self.cy = det[:,1]
        self.pol = det[:,2]
        self.mask = det[:,3].astype('u1')
        self.size = 2*int(np.ceil(np.sqrt(self.cx*self.cx + self.cy*self.cy).max())) + 1
        
        norm = np.sqrt(self.cx**2 + self.cy**2 + self.detd**2)
        self.qx = 500. * self.cx / norm
        self.qy = 500. * self.cy / norm
        self.qz = 500. * (self.detd / norm - 1.)

        self.dx = self.dy = self.size/2

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
            self.rz_embed = np.zeros((1001,1001))
            weights = np.zeros_like(self.rz_embed)
            np.add.at(weights, [R+500, Z+500], 1)
            np.add.at(self.rz_embed, [R+500, Z+500], self.image[self.mask<2])
            if self.r_avg_flag:
                weights = weights + weights[::-1]
                self.rz_embed = self.rz_embed + self.rz_embed[::-1]
            self.rz_embed[weights>0] /= weights[weights>0]

            self.imview.setImage(self.rz_embed, autoLevels=False, autoRange=self.auto_range, autoHistogramRange=False)
            self.auto_range = False
        else:
            rx = np.cos(self.phi)*self.cx - np.sin(self.phi)*self.cy
            ry = np.cos(self.phi)*self.cy + np.sin(self.phi)*self.cx
            rx = np.round(rx).astype('i4')[self.mask<2]
            ry = np.round(ry).astype('i4')[self.mask<2]
            self.rot_image = np.zeros((self.size, self.size))
            weights = np.zeros_like(self.rot_image)
            np.add.at(weights, [rx+self.dx, ry+self.dy], 1)
            weights[weights==0] = 1
            np.add.at(self.rot_image, [rx+self.dx, ry+self.dy], self.image[self.mask<2])
            self.rot_image /= weights
            self.imview.setImage(self.rot_image, autoLevels=False, autoRange=self.auto_range, autoHistogramRange=False)
            self.auto_range = False

    def phi_changed(self, value=None):
        self.phi = np.pi * value / 180.
        self.phi_val.setText('%.3d'%value)

    def beta_changed(self, value=None):
        self.beta = np.pi * value / 180. / 2.
        self.beta_val.setText('%4.1f'%(value/2.))

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

    def fname_changed(self, text=None):
        self.h5_fname = text
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
            print 'Saving to', fname
            np.save(fname, self.rz_embed)
        else:
            fname = os.path.splitext(self.h5_fname)[0]+'_'+'%.3d'%self.frame_num+'.npy'
            print 'Saving to', fname
            np.save(fname, self.rot_image)

    def keyPressEvent(self, event=None):
        if event.key() == QtCore.Qt.Key_Return:
            self.replot()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    #gui = RZ_gui('allruns_TMV_outstandingmulti_cheetahformat.h5')
    #gui = RZ_gui('tmv_best.h5')
    gui = RZ_gui('TMV_outstanding.h5', 'data/calib')
    sys.exit(app.exec_())
