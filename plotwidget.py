# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar, FigureCanvasQTAgg as FigureCanvas


class PlotWidget(QtWidgets.QWidget):
    """FigureCanvas & NavigationToolbar"""
    def __init__(self, parent=None, flags=QtCore.Qt.WindowFlags()):
        super(PlotWidget, self).__init__(parent, flags)
        layout = QtWidgets.QVBoxLayout(self)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.toolbar = NavigationToolbar(self.canvas, parent=self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

