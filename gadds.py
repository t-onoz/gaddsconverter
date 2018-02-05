#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from logging import getLogger

import fabio
import numpy as np
from fabio.brukerimage import BrukerImage
from numpy import sin, cos, tan, arcsin, arccos, arctan
from scipy.interpolate import RegularGridInterpolator

logger = getLogger(__name__)

DETTYPE = {
    # DETECTORNAME: (PIXPERCM, CMTOGRID)
    'MULTIWIRE': (47.5, 2.0),
    'CCD-PXL-2K': (56.02, 0.8),
    'CCD-PXL-ARR': (32.0, 0.4),
    'CCD-PXL-KAF1500': (51.2, 0.8),
    'CCD-PXL-L6000N': (56.02, 0.8),
    'CCD-PXL-L6000F': (56.02, 0.8),
    'CCD-PXL-L6500': (32.00, 1.5),
    'CCD-PXL-KAF2': (81.92, 0.8),
    'CCD-PXL-KAF': (81.92, 0.8),
    'CCD-PXL-MSPD': (81.92, 0.8),
    'CCD-PXL': (81.92, 0.8),
    'CCD-PXL-LDI': (83.333, 0.0),
}

if not (b'FORMAT :100', 'bruker100') in fabio.openimage.MAGIC_NUMBERS:
    fabio.openimage.MAGIC_NUMBERS.append((b'FORMAT :100', 'bruker100'))


class AreaDetectorImage(object):
    def __init__(self, image=None):
        """
        :param image: filename, file-like, or fabio image object
        """
        if image is None:
            image = fabio.fabioimage.FabioImage()
        elif not isinstance(image, fabio.fabioimage.FabioImage):
            image = fabio.open(image)
        if image.data is not None:
            if np.issubdtype(image.data.dtype, np.integer) and np.min(image.data) >= 0:
                image.data = image.data.astype(np.min_scalar_type(np.max(image.data)))
        self.image = image
        self.alpha = np.nan  # 検出器の中心の2θ値 (単位：radian)
        self.distance = np.nan  # 試料から検出面までの距離（単位：cm）
        self.densityXY = (np.nan, np.nan)  # 単位長さ (cm) 辺りのピクセル数
        self.centerXY = (np.nan, np.nan)  # 検出器中心のx, y座標（単位：ピクセル）。
        self.scale = 1  # linear scale factor
        self.offset = 0  # linear offset
        self.limits = (np.nan, np.nan, np.nan, np.nan)  # min2θ, max2θ, minγ, maxγ
        self.data_converted = np.ndarray((0, 0), dtype=int)
        self.indexes = (np.arange(0), np.arange(0))
        self.load_headers()

    def xy_to_angles(self, x, y):
        """convert from (x, y) coordinates to (twoth, gamma).
         (x, y) = (0, 0) corresponds to the detector center, and they should be given in cm.
         
        Reference
        B.B. He, Two-Dimensional X-Ray Diffraction (Wiley, 2011). 
        2.3.4 Pixel Position in Diffraction Space -- Flat Detector

        :param x: x coordinate in cm.
        :param y: y coordinate in cm
        :return: (twoth, gamma) in rad.
        """
        alpha = self.alpha
        D = self.distance

        twoth = arccos((x*sin(alpha) + D*cos(alpha))/np.sqrt(D**2 + x**2 + y**2))

        det = x*cos(alpha) - D*sin(alpha)
        sign = ((det >= 0) - 0.5)/0.5
        gamma = sign * arccos(-y / np.sqrt(y**2 + det**2))

        return twoth, gamma

    def rowcol_to_angles(self, row, col):
        # row ↔ y, col ↔ x
        dX, dY = self.densityXY
        cX, cY = self.centerXY
        nY, nX = self.image.data.shape
        x, y = (col - cX)/dX, -(row-(nY-cY))/dY
        return self.xy_to_angles(x, y)

    def angles_to_rowcol(self, twoth, gamma):
        """
        convert from (twoth, gamma) to (row, col).
        angles are given in rad.
                
        Reference
        B.B. He, Two-Dimensional X-Ray Diffraction (Wiley, 2011). 
        2.3.4 Pixel Position in Diffraction Space -- Flat Detector

        """
        # 2θ=90°でのtan(2θ)の発散対策
        if np.isscalar(twoth):
            if np.isclose(twoth, np.pi/2, atol=1e-6, rtol=0):
                twoth += 1e-6
        else:
            twoth = np.array(twoth, dtype=float)
            twoth[np.isclose(twoth, np.pi/2, atol=1e-6, rtol=0)] += 1e-6

        alpha = self.alpha % (2 * np.pi)
        D = self.distance

        x = D * (cos(alpha)*tan(twoth)*sin(gamma) + sin(alpha)) / \
            (cos(alpha) - sin(alpha)*tan(twoth)*sin(gamma))
        y = -(x*sin(alpha) + D*cos(alpha)) * tan(twoth) * cos(gamma)

        # (x, y) = (0, 0) は図の左下に、(row, col) = (0, 0) は左上に対応する。
        # x と col、yとrowがそれぞれ対応する（とてもややこしい）。
        return (self.image.data.shape[0] - y * self.densityXY[1] - self.centerXY[1],
                x * self.densityXY[0] + self.centerXY[0])

    def relim(self):
        if self.image.data is None:
            return
        rr, cc = np.indices(self.image.data.shape)
        twoth, gamma = self.rowcol_to_angles(rr, cc)
        self.limits = (np.min(twoth), np.max(twoth), np.min(gamma), np.max(gamma))

    def convert(self, n_twoth=None, n_gamma=None):
        if n_twoth is None:
            n_twoth = self.image.data.shape[1]
        if n_gamma is None:
            n_gamma = self.image.data.shape[0]

        # determine range of twoth and gamma
        self.relim()
        seq_twoth = np.linspace(self.limits[0], self.limits[1], n_twoth)
        if self.alpha >= 0:
            seq_gamma = np.linspace(self.limits[2], self.limits[3], n_gamma)
        else:
            seq_gamma = np.linspace(self.limits[3], self.limits[2], n_gamma)
        self.indexes = tuple(np.rad2deg((seq_gamma, seq_twoth)))

        # create regular (twoth, gamma) grid and then convert it to (row, col)
        newrow, newcol = self.angles_to_rowcol(*np.meshgrid(seq_twoth, seq_gamma, indexing='xy'))

        # perform interpolation
        f = RegularGridInterpolator(
            (np.arange(self.image.data.shape[0]), np.arange(self.image.data.shape[1])),
            self.image.data,
            method='nearest',
            bounds_error=False,
            fill_value=0
         )
        new = f(np.c_[newrow.ravel(), newcol.ravel()]).reshape((n_gamma, n_twoth)).astype(self.image.data.dtype)
        if self.scale != 1 or self.offset != 0:
            new = new.astype(np.float32) * self.scale + self.offset
        self.data_converted = new

    def gridline(self, angle_deg, axis='twoth', delta_deg=0.1):
        """GADDS image上にプロットできる等2θ, 等γのグリッド線のデータを取り出す。

        Note: self.alphaやself.distanceを手動で変更した場合、gridlineの前にself.relim()を実行する必要がある。"""
        angle = np.deg2rad(angle_deg)
        delta = np.deg2rad(delta_deg)
        if axis == 'twoth':
            if not self.limits[0] <= angle <= self.limits[1]:
                return [], []
            rows, cols = self.angles_to_rowcol(angle, np.arange(self.limits[2], self.limits[3], delta))
        elif axis == 'gamma':
            if not self.limits[2] <= angle <= self.limits[3]:
                return [], []
            rows, cols = self.angles_to_rowcol(np.arange(self.limits[0], self.limits[1], delta), angle)
        else:
            raise ValueError('unknown axis: %s' % axis)
        idx = (0 <= rows) & (rows < self.image.dim2) & (0 <= cols) & (cols < self.image.dim1)
        return cols[idx], rows[idx]

    def load_headers(self):
        """image fileのヘッダーから必要なパラメーターを読み込む"""
        image = self.image
        if isinstance(image, BrukerImage):
            """gfrmファイルにはversion 86と100がある（メモ帳で開いた先頭部分で確認可能）。
            version 86についてはGADDS付属のgaddsref.pdfに詳しく説明があるが、100の場合は説明がない。
            ここでは、DIFFRAC.EVAで開いた時に表示される値から推察されるパラメーターを設定している。
            """
            if 'UNWARPED' not in image.header['TYPE']:
                logger.warning('This frame has NOT been UNWARPED (corrected), and may contain some error.', stack_info=True)
            self.alpha = np.deg2rad(float(image.header['ANGLES'].split()[0]))

            # CENTER
            # ver 86: x, yの2つの値が記録されている。
            # ver100: 値が4つあるが、後半の2つが実際に使われているようだ。
            self.centerXY = tuple(float(x) for x in image.header['CENTER'].split()[-2:])

            # linear scale and offset
            if 'LINEAR' in image.header:
                self.scale, self.offset = (float(x) for x in image.header['LINEAR'].split()[:2])

            # PIXPERCM: 1cm辺りのピクセル数（フレームが512x512の場合）
            m = re.search('PIXPERCM:([0-9\.]+)', image.header['DETTYPE'])
            if m:
                pixpercm = float(m.groups()[0])
            else:
                try:
                    # version 100の場合、DETTYPEの書式は 「名前 pixpercm cmtogrid （以降不明）」になるようだ。
                    pixpercm = float(image.header['DETTYPE'].split()[1])
                except (ValueError, IndexError):
                    pixpercm = DETTYPE[image.header['DETTYPE']][0]
            nrows, ncols = image.data.shape
            self.densityXY = (pixpercm * ncols / 512, pixpercm * nrows / 512)

            # detector distance（試料と検出面の距離）
            # DISTANCフィールドの一つ目の数字に、CMTOGRIDの値を加えたもの。
            distanc = float(image.header['DISTANC'].split()[0])
            m = re.search('CMTOGRID:([0-9\.]+)', image.header['DETTYPE'])
            if m:
                cmtogrid = float(m.groups()[0])
            else:
                try:
                    cmtogrid = float(image.header['DETTYPE'].split()[2])
                except (ValueError, IndexError):
                    cmtogrid = DETTYPE[image.header['DETTYPE']][1]
            self.distance = distanc + cmtogrid
        else:
            pass
        self.relim()


if __name__ == '__main__':
    # usage example
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import sys
    try:
        f = sys.argv[1]
    except IndexError:
        raise ValueError('please specify filename.')
    areaimage = AreaDetectorImage(f)
    areaimage.convert()
    matrix_original = areaimage.image.data

    dx = areaimage.indexes[1][1] - areaimage.indexes[1][0]
    dy = areaimage.indexes[0][1] - areaimage.indexes[0][0]
    extent = (
        areaimage.indexes[1][0]-dx/2, areaimage.indexes[1][-1]+dx/2,
        areaimage.indexes[0][-1]-dy/2, areaimage.indexes[0][0]+dy/2
    )
    im = plt.imshow(areaimage.data_converted,
                interpolation='nearest',
                vmin=0, vmax=10,
                aspect='auto',
                origin='upper',
                extent=extent)
    plt.colorbar()
    plt.show()
