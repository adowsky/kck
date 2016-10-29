#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division             # Division in Python 2.7
import matplotlib
matplotlib.use('Agg')                       # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from matplotlib import colors

    
def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    # rc('text', usetex=True)
    # rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400         # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)

    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')


def hsv2rgb(h, s, v):
    if v == 0:
        r = 0
        g = 0
        b = 0
    elif h <= 60:
        b = 1 - s
        r = v
        g = h/60 * v
    elif h <= 120:
        b = 1 - s
        g = v
        r = (1 - (h - 60)/60) * v
    elif h <= 180:
        r = 1 - s
        g = v
        b = (h - 120)/60 * v
    elif h <= 240:
        r = 1 - s
        b = v
        g = (1 - (h - 180))/60 * v
    elif h <= 300:
        g = 1 - s
        b = v
        r = (h - 240)/60 * v
    else:
        g = 1 - s
        r = v
        b = (1 - (h - 300))/60 * v

    if r >= 1 and b > 0:
        print(r)

    return r, g, b


def gradient_rgb_bw(v):
    return v, v, v


def gradient_rgb_gbr(v):
    if v > 0.5:
        v2 = (1 - v)*2
    else:
        v2 = v*2
    return max(0, (v - 0.5)*2), max(0, (0.5 - v)*2), v2


def gradient_rgb_gbr_full(v):
    if v <= 0.5:
        b = v*4
    else:
        b = (1-v)*4
    return min(1, max((4*v - 2), 0)), min(1, max((-4*v + 2), 0)), min(1, b)


def gradient_rgb_wb_custom(v):
    if v <= 3/7:
        r = max(0, min(1, -7*v + 2))
    elif v <= 10/14:
        r = max(0, min(1, 7*v - 4))
    else:
        r = max(0, min(1, -7*v + 7))

    #green
    if v <= 2/7:
        g = max(0, -7*v + 1)
    elif v <= 4/7:
        g = max(0, min(1, 7*v - 2))
    else:
        g = max(0, min(1, -7*v + 6))

    return r, g, max(0, min(1, -7*v + 4))


def gradient_hsv_bw(v):
    
    return hsv2rgb(0, v, 1)


def gradient_hsv_gbr(v):
    #TODO
    return hsv2rgb(0, 0, 0)

def gradient_hsv_unknown(v):
    #TODO
    return hsv2rgb(v, 0, 0)


def gradient_hsv_custom(v):
    #TODO
    return hsv2rgb(0, 0, 0)


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])

