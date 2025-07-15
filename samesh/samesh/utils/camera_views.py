#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

# ------------------------------------------------------------------------------
#                                视角定义
# ------------------------------------------------------------------------------
distance = 2.5
VIEW_POSITIONS = {
    "top-left": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(math.pi/4),
                     distance * math.sin(math.pi/4),
                     distance * math.cos(math.pi/4) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "left": {
        "position": [-distance * math.cos(0) * math.sin(math.pi/4),
                     distance * math.sin(0),
                     distance * math.cos(0) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "bottom-left": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(math.pi/4),
                     distance * math.sin(-math.pi/4),
                     distance * math.cos(-math.pi/4) * math.cos(math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "top-center": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(0),
                     distance * math.sin(math.pi/4),
                     distance * math.cos(math.pi/4) * math.cos(0)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "center": {
        "position": [0.0, 0.0, distance],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "bottom-center": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(0),
                     distance * math.sin(-math.pi/4),
                     distance * math.cos(-math.pi/4) * math.cos(0)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "top-right": {
        "position": [-distance * math.cos(math.pi/4) * math.sin(-math.pi/4),
                     distance * math.sin(math.pi/4),
                     distance * math.cos(math.pi/4) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "right": {
        "position": [-distance * math.cos(0) * math.sin(-math.pi/4),
                     distance * math.sin(0),
                     distance * math.cos(0) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "bottom-right": {
        "position": [-distance * math.cos(-math.pi/4) * math.sin(-math.pi/4),
                     distance * math.sin(-math.pi/4),
                     distance * math.cos(-math.pi/4) * math.cos(-math.pi/4)],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    },
    "custom": {
        "position": [0.0, 0.0, distance],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0]
    }
} 