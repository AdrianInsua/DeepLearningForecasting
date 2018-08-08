# -*- coding: utf-8 -*-
"""This scripts offer method to analyce principal components of collection of elements.

Methods:
    * reduceDimensionality
    * getPCAComponents
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt

class PcaAnalysis:
    """Principal component class"""
    def __init__(self, v, data):
        self.data = data
        self.v = v
        self.pca_omponents = None
        self.var = None
        self.cum_var = None

    def reduce_dimensionality(self, n_components):
        """Reduce dimensionality"""
        pcan = PCA(n_components)
        pcan.fit(self.data)
        return pcan.fit_transform(self.data)

    def get_pca_components(self, n_components):
        """Get principal component analysis"""
        self.data = scale(self.data)
        pcan = PCA(n_components)
        pcan.fit(self.data)
        self.var = pcan.explained_variance_ratio_
        self.cum_var = np.cumsum(np.round(pcan.explained_variance_ratio_, decimals=4)*100)

    def show_analysis(self):
        """Plot analysis"""
        plt.plot(self.var)
        plt.plot(self.cum_var)
