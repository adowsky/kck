#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division  # Division in Python 2.7
import matplotlib

matplotlib.use('Agg')  # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from matplotlib import colors


def left_neighbour_shadow(line):
    result = [hsv2rgb(120 - line[0]*120, 0.5, 0.5)]
    for i in range(len(line[1:])):
        value = 0.6 + (line[i + 1] - line[i]) / 0.6
        result.append(hsv2rgb(120 - line[i+1]*120, 0.7, value))
    return result


def plot_map(filename):
    rc('legend', fontsize=10)
    w = 0
    h = 0
    pt_per_inch = 0
    data = []

    with open(filename) as file:
        lines = file.readlines()
        initial_data = lines[0].split(" ")
        h = int(initial_data[0])
        w = int(initial_data[1])
        pt_per_inch = int(initial_data[2])
        print("processing lines of file...")
        for line in lines[1:]:
            splitted = line.strip().split(" ")
            for x in splitted:
                if float(x)/160 > 1:
                    print(x)
            data.append(left_neighbour_shadow([float(x)/160 for x in splitted]))
        print("processing file: [DONE]")
    im = plt.imshow(data, aspect='auto')
    plt.savefig('map.pdf')


def plot_color_gradients(gradients, names):
    # For pretty latex fonts (commented out, because it does not work on some machines)
    # rc('text', usetex=True)
    # rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400  # Show in latex using \the\linewidth
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
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')


def hsv2rgb(h, s, v):
    if v == 0:
        r = 0
        g = 0
        b = 0
    elif h <= 60:
        b = (1 - s) * v
        r = v
        g = b + h / 60 * s * v
    elif h <= 120:
        b = (1 - s) * v
        g = v
        r = b + ((120 - h) / 60) * s * v
        o = 0
    elif h <= 180:
        r = (1 - s) * v
        g = v
        b = r + (h - 120) / 60 * s * v

    elif h <= 240:
        r = (1 - s) * v
        b = v
        g = r + ((240 - h) / 60) * s * v
        o = 0
    elif h <= 300:
        g = (1 - s) * v
        b = v
        r = g + (h - 240) / 60 * s * v
        o = 0
    else:
        g = (1 - s) * v
        r = v
        b = g + ((360 - h) / 60) * s * v

    return r, g, b


def gradient_rgb_bw(v):
    return v, v, v


def gradient_rgb_gbr(v):
    if v > 0.5:
        v2 = (1 - v) * 2
    else:
        v2 = v * 2
    return max(0, (v - 0.5) * 2), max(0, (0.5 - v) * 2), v2


def gradient_rgb_gbr_full(v):
    if v <= 0.5:
        b = v * 4
    else:
        b = (1 - v) * 4
    return min(1, max((4 * v - 2), 0)), min(1, max((-4 * v + 2), 0)), min(1, b)


def gradient_rgb_wb_custom(v):
    if v <= 3 / 7:
        r = max(0, min(1, -7 * v + 2))
    elif v <= 10 / 14:
        r = max(0, min(1, 7 * v - 4))
    else:
        r = max(0, min(1, -7 * v + 7))

    # green
    if v <= 2 / 7:
        g = max(0, -7 * v + 1)
    elif v <= 4 / 7:
        g = max(0, min(1, 7 * v - 2))
    else:
        g = max(0, min(1, -7 * v + 6))

    return r, g, max(0, min(1, -7 * v + 4))


def gradient_hsv_bw(v):
    return hsv2rgb(0, 0, v)


def gradient_hsv_gbr(v):
    return hsv2rgb(120 + v * 240, 1, 1)


def gradient_hsv_unknown(v):
    return hsv2rgb(120 - v * 120, 0.5, 1)


def gradient_hsv_custom(v):
    return hsv2rgb(v*360, 1 - v, 1)


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()


    #gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
     #        gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

   # plot_color_gradients(gradients, [toname(g) for g in gradients])
    plot_map("big.dem")
