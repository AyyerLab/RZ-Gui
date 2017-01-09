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
        self.z_avg_flag = False
        self.subtract_bg_flag = False
        self.auto_range = True
        self.frame_changed = True
        self.assem_shape = None
        self.assem = None
        self.angles = None
        self.phi = 0.
        self.beta = 0.
        self.detd = 90. / 0.11
        
        self.get_geom()
        
        self.init_UI()

    def init_UI(self):
        # Overall layout
        window = QtGui.QVBoxLayout()

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
        self.phi_slider.setRange(0,720)
        hbox.addWidget(self.phi_slider)
        self.phi_val = QtGui.QLabel('%4.1f'%(self.phi_slider.value()/2.), self)
        hbox.addWidget(self.phi_val)
        label = QtGui.QLabel('    Beta', self)
        hbox.addWidget(label)
        self.beta_slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.beta_slider.valueChanged.connect(self.beta_changed)
        self.beta_slider.sliderReleased.connect(self.replot)
        self.beta_slider.setRange(0,180)
        hbox.addWidget(self.beta_slider)
        self.beta_val = QtGui.QLabel('%4.1f'%(self.beta_slider.value()/2.), self)
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
        self.r_avg_button = QtGui.QCheckBox('Meridional average', self)
        self.r_avg_button.stateChanged.connect(self.r_avg_flag_changed)
        self.r_avg_button.setChecked(False)
        hbox.addWidget(self.r_avg_button)
        self.z_avg_button = QtGui.QCheckBox('Equatorial average', self)
        self.z_avg_button.stateChanged.connect(self.z_avg_flag_changed)
        self.z_avg_button.setChecked(False)
        hbox.addWidget(self.z_avg_button)
        self.subtract_bg_button = QtGui.QCheckBox('Subtract BG', self)
        self.subtract_bg_button.stateChanged.connect(self.subtract_bg_flag_changed)
        self.subtract_bg_button.setChecked(False)
        hbox.addWidget(self.subtract_bg_button)
        hbox.addStretch(1)
        button = QtGui.QPushButton('Save Angles', self)
        button.clicked.connect(self.save_angles)
        hbox.addWidget(button)
        button = QtGui.QPushButton('Save', self)
        button.clicked.connect(self.save_image)
        hbox.addWidget(button)
        button = QtGui.QPushButton('Save All', self)
        button.clicked.connect(self.save_all)
        hbox.addWidget(button)
        button = QtGui.QPushButton('Quit', self)
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
                    print 'Reading angles from', fname
                    self.angles = np.loadtxt(fname)
                else:
                    self.angles = np.zeros((f[self.h5_dset].shape[0], 3))
                    self.angles[:,0] = np.arange(self.angles.shape[0])
         
        self.phi_slider.setValue(self.angles[self.frame_num, 1]*2.)
        self.beta_slider.setValue(self.angles[self.frame_num, 2]*2.)
        
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
        x, y = np.indices((1001,1001))
        x -= 500; y -= 500
        self.intrad = np.sqrt(x*x + y*y).astype('i4')
        self.radpix = np.ones_like(self.intrad, dtype=np.bool)
        # ---- Amyloid ----
        self.radpix[np.absolute(y)>120] = False
        self.radpix[np.absolute(y)<40] = False
        self.radpix[(self.intrad<=40) & (np.absolute(y)>20)] = True
        # -----------------
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
            self.rz_embed = np.zeros((1001,1001))
            self.weights = np.zeros_like(self.rz_embed)
            np.add.at(self.weights, [R+500, Z+500], 1)
            np.add.at(self.rz_embed, [R+500, Z+500], self.image[self.mask<2])
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
            print 'Saving to', fname
            np.save(fname, self.rz_embed)
            fname = os.path.splitext(self.h5_fname)[0]+'_'+'%.3d'%self.frame_num+'_rzw.npy'
            np.save(fname, self.weights)
        else:
            fname = os.path.splitext(self.h5_fname)[0]+'_'+'%.3d'%self.frame_num+'.npy'
            print 'Saving to', fname
            np.save(fname, self.rot_image)

    def save_angles(self):
        fname = os.path.splitext(self.h5_fname)[0]+'_angles.dat'
        print 'Saving angles to', fname
        np.savetxt(fname, self.angles, fmt='%.4d %5.1f %5.1f', header='Num    Phi  Beta', comments='')

    def save_all(self):
        for i in range(self.num_frames):
            self.frame_num = i
            self.frame_changed = True
            self.replot()
            self.save_image()

    def keypressevent(self, event=None):
        if event.key() == QtCore.Qt.Key_return:
            self.replot()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    if len(sys.argv) > 1:
        gui = RZ_gui(sys.argv[1], 'data/calib')
    else:
        gui = RZ_gui('bombesin_well_oriented.h5', 'data/calib')
    sys.exit(app.exec_())
