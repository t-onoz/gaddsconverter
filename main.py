#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
from os import path
from logging import getLogger
import numpy as np
from PyQt5 import QtWidgets, QtCore
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ibwwriter import IgorBinaryWave
from gadds import AreaDetectorImage
from ui_mainwindow import Ui_MainWindow
# required for pyinstaller
import scipy
import scipy.interpolate
logger = getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    fileLoaded = QtCore.pyqtSignal()
    convertFinished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pbOpen.clicked.connect(self.open_file)
        self.ui.pbConvert.clicked.connect(self.convert)
        self.ui.pbSaveOriginal.clicked.connect(self.save_original)
        self.ui.pbSaveConverted.clicked.connect(self.save_converted)
        self.ui.pbSaveGrids.clicked.connect(self.save_grids)
        self.gfrm = AreaDetectorImage()
        self.fileLoaded.connect(lambda: self.ui.pbConvert.setEnabled(True))
        self.fileLoaded.connect(lambda: self.ui.pbSaveGrids.setEnabled(True))
        self.fileLoaded.connect(lambda : self.ui.pbSaveOriginal.setEnabled(True))
        self.fileLoaded.connect(lambda: self.ui.pbSaveConverted.setEnabled(False))
        self.fileLoaded.connect(lambda: self.ui.tabPlots.setCurrentIndex(0))
        self.fileLoaded.connect(self.plot_original)
        self.fileLoaded.connect(self.show_parameters)
        self.convertFinished.connect(lambda: self.ui.pbSaveConverted.setEnabled(True))
        self.convertFinished.connect(self.plot_converted)
        self.ui.cbGrid2th.stateChanged.connect(self.plot_gridlines)
        self.ui.sbGrid2th.valueChanged.connect(self.plot_gridlines)
        self.ui.cbGridGamma.stateChanged.connect(self.plot_gridlines)
        self.ui.sbGridGamma.valueChanged.connect(self.plot_gridlines)

    def open_file(self, *, f=None):
        if f is None:
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select file', '', '*.gfrm; *.sfrm;; *')
        if f:
            try:
                self.gfrm = AreaDetectorImage(f)
            except Exception:
                msg = 'could not open file: %s' % f
                logger.warning(msg, exc_info=True)
                QtWidgets.QMessageBox.warning(self, 'error', msg)
            else:
                self.fileLoaded.emit()
                self.setWindowTitle(f)

    def plot_original(self):
        data = self.gfrm.image.data
        if data is None:
            return
        if self.gfrm.scale != 1 or self.gfrm.offset != 0:
            data = data.astype(np.float32) * self.gfrm.scale + self.gfrm.offset
        xlim = (0, self.gfrm.image.data.shape[1])
        ylim = (self.gfrm.image.data.shape[0], 0)
        self.ui.plotOriginal.figure.clf()
        self.ui.plotOriginal.figure.set_tight_layout(True)
        ax = self.ui.plotOriginal.figure.add_subplot(111)
        ax.set_title('image')
        im = ax.imshow(data, cmap='afmhot', norm=colors.LogNorm(vmin=0.1), aspect=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale(False)
        plt.colorbar(mappable=im, ax=ax)
        self.plot_gridlines()
        self.ui.plotOriginal.canvas.draw()

    def plot_converted(self):
        self.ui.plotConverted.figure.clf()
        self.ui.plotConverted.figure.set_tight_layout(True)
        ax = self.ui.plotConverted.figure.add_subplot(111)
        ax.set_title('converted image')
        im = ax.imshow(self.gfrm.data_converted,
                       interpolation='nearest',
                       norm=colors.LogNorm(vmin=0.1),
                       aspect='auto',
                       origin='upper',
                       cmap='jet')
        dx = self.gfrm.indexes[1][1] - self.gfrm.indexes[1][0]
        dy = self.gfrm.indexes[0][1] - self.gfrm.indexes[0][0]
        extent = (
            self.gfrm.indexes[1][0]-dx/2, self.gfrm.indexes[1][-1]+dx/2,
            self.gfrm.indexes[0][-1]-dy/2, self.gfrm.indexes[0][0]+dy/2
        )
        im.set_extent(extent)
        ax.set_xticks(np.arange(0, 180, 10))
        ax.set_yticks(np.arange(-170, 190, 20))
        ax.set_xlim(im.get_extent()[:2])
        ax.set_ylim(im.get_extent()[2:])
        ax.set_xlabel(r'$2\theta$ (°)')
        ax.set_ylabel(r'$\gamma$ (°)')
        ax.autoscale(False)
        plt.colorbar(mappable=im, ax=ax)
        ax.grid(True, color='#aaaaaa', ls=':', lw=0.5)
        self.ui.plotConverted.canvas.draw()
        self.ui.tabPlots.setCurrentWidget(self.ui.plotConverted)

    def plot_gridlines(self, *, ax=None, clear_previous=True):
        if ax is None:
            if not self.ui.plotOriginal.figure.axes:
                return
            ax = self.ui.plotOriginal.figure.axes[0]
        if clear_previous:
            del ax.lines[:]
        if self.ui.cbGrid2th.isChecked():
            delta_deg = self.ui.sbGrid2th.value()
            for angle_deg in np.arange(delta_deg, 180, delta_deg):
                if self.gfrm.limits[0] <= np.deg2rad(angle_deg) <= self.gfrm.limits[1]:
                    ax.plot(*self.gfrm.gridline(angle_deg, 'twoth'), ':', lw=1, color='#cccccc')
        if self.ui.cbGridGamma.isChecked():
            delta_deg = self.ui.sbGridGamma.value()
            for angle_deg in np.concatenate((np.arange(-90, 180, delta_deg), np.arange(-90-delta_deg, -180, -delta_deg))):
                if self.gfrm.limits[2] <= np.deg2rad(angle_deg) <= self.gfrm.limits[3]:
                    ax.plot(*self.gfrm.gridline(angle_deg, 'gamma'), ':', lw=1, color='#cccccc')
        self.ui.plotOriginal.canvas.draw()

    def show_parameters(self):
        text = '%s = %s\n' % ('alpha', np.rad2deg(self.gfrm.alpha))
        for p in ('distance', 'densityXY', 'centerXY', 'scale', 'offset'):
            text += '%s = %s\n' % (p, getattr(self.gfrm, p))
        text += '2θ: (%.2f, %.2f)\n' % tuple(np.rad2deg(self.gfrm.limits[:2]))
        text += 'γ: (%.2f, %.2f)\n' % tuple(np.rad2deg(self.gfrm.limits[2:]))
        self.ui.teParameters.setPlainText(text)

    def convert(self):
        try:
            self.gfrm.convert()
        except Exception:
            msg = 'convertion failed!'
            logger.warning(msg, exc_info=True)
            QtWidgets.QMessageBox.warning(self, 'error', msg)
        else:
            self.convertFinished.emit()

    def save_converted(self, *, f=None):
        if f is None:
            f, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'select file', '', '*.ibw;; *')
        if f:
            ibw = IgorBinaryWave(self.gfrm.data_converted, name='converted')
            x0 = self.gfrm.indexes[1][0]
            dx = self.gfrm.indexes[1][1] - self.gfrm.indexes[1][0]
            y0 = self.gfrm.indexes[0][0]
            dy = self.gfrm.indexes[0][1] - self.gfrm.indexes[0][0]
            ibw.set_dimscale('x', x0, dx)
            ibw.set_dimscale('y', y0, dy)
            opener = gzip.open if f.endswith('.gz') else open
            with opener(f, 'wb') as fp:
                ibw.save(fp, image=True)

    def save_original(self, *, f=None):
        if f is None:
            f, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'select file', '', '*.ibw;; *')
        if f:
            if self.gfrm.scale != 1 or self.gfrm.offset != 0:
                data = self.gfrm.image.data.astype(np.float32) * self.gfrm.scale + self.gfrm.offset
            else:
                data = self.gfrm.image.data
            ibw = IgorBinaryWave(data, name='original')
            opener = gzip.open if f.endswith('.gz') else open
            with opener(f, 'wb') as fp:
                ibw.save(fp, image=True)

    def save_grids(self, *, f=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(6, 6)
        self.plot_gridlines(ax=ax)
        if not ax.lines:
            QtWidgets.QMessageBox.warning(self, 'error', 'no grid (twoth, gamma) is checked.')
            return
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.gfrm.image.data.shape[1])
        ax.set_ylim(self.gfrm.image.data.shape[1], 0)
        if f is None:
            f, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'select file', '', '*.png;;*')
        if not f:
            return
        fig.savefig(f, dpi=300, transparent=True)
        gridfile = path.splitext(f)[0] + '_griddata.txt'
        with open(gridfile, 'wb') as fp:
            fp.write(b'grid_x\tgrid_y\n')
            for line in ax.lines:
                data = line.get_xydata()
                if data.size == 0:
                    continue
                np.savetxt(fp, data, delimiter='\t')

if __name__ == '__main__':
    import argparse
    import warnings
    for name in plt.colormaps():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmap = plt.get_cmap(name)
        cmap.set_bad(cmap(0))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='command line mode (batch operation)', action='store_true')
    parser.add_argument('files', nargs='*')
    args = parser.parse_args()

    app = QtWidgets.QApplication([])
    m = MainWindow()
    if args.batch:
        print('---------- batch mode ----------')
        if not args.files:
            raise ValueError('no files specified')
        for file in args.files:
            print('start file: %s' % file)
            noext = path.splitext(file)[0]
            m.open_file(f=file)
            m.convert()
            m.save_original(f=(noext + '_original.ibw.gz'))
            m.save_converted(f=(noext + '_converted.ibw.gz'))
            m.save_grids(f=(noext + '_grid.png'))
            m.ui.plotOriginal.figure.savefig(noext + '_original.png', dpi=300)
            m.ui.plotConverted.figure.savefig(noext + '_converted.png', dpi=300)
    else:
        m.show()
        app.exec_()
