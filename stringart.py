#!/bin/python

# -*- coding: utf-8 -*-
"""stringart.py - A program to calculate and visualize string art

Some more information will follow
"""

import math
import copy
import numpy as np
import bresenham

from PIL import Image, ImageOps, ImageFilter, ImageEnhance


class StringArtGenerator:
    def __init__(self):
        self.iterations = 1000
        self.image = None
        self.data = None
        self.residual = None
        self.seed = 0
        self.nails = 100
        self.weight = 20
        self.nodes = []
        self.paths = []

    def set_seed(self, seed):
        self.seed = seed

    def set_weight(self, weight):
        self.weight = weight

    def set_nails(self, nails):
        self.nails = nails
        self.set_nodes()

    def set_iterations(self, iterations):
        self.iterations = iterations

    def set_nodes(self):
        """
        Sets nails evenly along a circle of given diameter
        """

        spacing = (2*math.pi)/self.nails

        steps = range(self.nails)

        radius = self.get_radius()

        x = [radius + radius*math.cos(t*spacing) for t in steps]
        y = [radius + radius*math.sin(t*spacing) for t in steps]

        self.nodes = list(zip(x, y))

    def get_radius(self):
        return 0.5*np.amax(np.shape(self.data))

    def load_image(self, path):
        img = Image.open(path)
        self.image = img
        np_img = np.array(self.image)
        self.data = np.flipud(np_img).transpose()

    def preprocess(self):
        # Convert image to grayscale
        self.image = ImageOps.grayscale(self.image)
        self.image = ImageOps.invert(self.image)
        self.image = self.image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        self.image = ImageEnhance.Contrast(self.image).enhance(1)
        np_img = np.array(self.image)
        self.data = np.flipud(np_img).transpose()

    def generate(self):
        self.calculate_paths()
        pattern = []
        nail = self.seed
        datacopy = copy.deepcopy(self.data)
        for i in range(self.iterations):
            # calculate straight line to all other nodes and calculate
            # 'darkness' from start node

            # choose max darkness path
            darkest_nail, darkest_path = self.choose_darkest_path(nail)

            # add chosen node to pattern
            pattern.append(self.nodes[darkest_nail])

            # substract chosen path from image
            self.data = self.data - self.weight*darkest_path
            self.data[self.data < 0.0] = 0.0

            if (np.sum(self.data) <= 0.0):
                print("Stopping iterations. No more data or residual unchanged.")
                break

            # store current residual as delta for next iteration
            delta = np.sum(self.data)

            # continue from destination node as new start
            nail = darkest_nail

        self.residual = copy.deepcopy(self.data)
        self.data = datacopy

        return pattern

    def generate_stepwise(self):
        """Generate pattern step by step and yield progress."""
        self.data = self.data.astype(
            np.float64)  # Ensure compatible type for subtraction
        self.calculate_paths()
        pattern = []
        nail = self.seed
        data_copy = copy.deepcopy(self.data)

        for _ in range(self.iterations):
            darkest_nail, darkest_path = self.choose_darkest_path(nail)
            pattern.append(self.nodes[darkest_nail])
            self.data -= self.weight * darkest_path
            self.data[self.data < 0] = 0  # Ensure no negative values
            if np.sum(self.data) == 0:
                break
            nail = darkest_nail
            yield self.nodes[darkest_nail]  # Yield progress

        self.data = data_copy  # Reset data after generation
        self.residual = copy.deepcopy(self.data)
        self.is_completed = True

    def choose_darkest_path(self, nail):
        max_darkness = -1.0
        for index, rowcol in enumerate(self.paths[nail]):
            rows = [i[0] for i in rowcol]
            cols = [i[1] for i in rowcol]
            darkness = float(np.sum(self.data[rows, cols]))

            if darkness > max_darkness:
                darkest_path = np.zeros(np.shape(self.data))
                darkest_path[rows, cols] = 1.0
                darkest_nail = index
                max_darkness = darkness

        return darkest_nail, darkest_path

    def calculate_paths(self):
        for nail, anode in enumerate(self.nodes):
            self.paths.append([])
            for node in self.nodes:
                path = bresenham.bresenham_path(anode, node, self.data.shape)
                self.paths[nail].append(path)
