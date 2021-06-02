#!/usr/bin/env python3
# Copyright: This document has been placed in the public domain.

"""
Grouped Taylor diagram implementation.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .taylorDiagram import TaylorDiagram

def group_taylor_diagram(model_cc,
                      model_std,
                      model_names=None,
                      model_desc='Models',
                      model_markers=None,
                      model_markersize=5,
                      show_model_legend=True,
                      gr_names=None,
                      gr_cmap='viridis',
                      gr_colors=None,
                      gr_desc='Groups',
                      show_gr_legend=True,
                      std_range=None,
                      *args,
                      **kwargs
                      ):
    """ Uses TaylorDiagram class to create Grouped Taylor Diagram.
    Different models are shown using differnt matplotlib markers and
    different groups are shown using colors.

    Parameters:

        * model_cc: Array of coefficient of correlation between different models
        		and reference model. First dimension is the number of
        		models and second dimension is the number of groups.
        * model_std: Standard deviation for data from different models. It should be
        		of same size as model_cc.
        * model_names: Name of Different models.
        * model_desc: Prefix for different models as shown in legends. Default is 'Models'.
        * model_markers: Array for valid matplotlib markers for models.
        * show_model_legend: Show/hide legend for model. Default is True (legend is shown).
        * model_markersize: Size of matplotlib marker being used for models.
        * gr_names: Name of groups
        * gr_cmap: valid matplotlib cmap for group colors. Default: viridis
        * gr_colors: array of colors. Default is None. If provided then gr_cmap is ignored.
        * gr_desc: Prefix for different groups as shown in legends. Default is 'Groups'.
        * show_gr_legend: Show/hide legend for group. Default is True (legend is shown).
        * std_range: Range of x axis (standard deviation range).

    Returns:

        * dia: TaylorDiagram instance
    """

    model_cc, model_std = np.asarray(model_cc), np.asarray(model_std)
    num_model, num_gr = model_cc.shape

    if gr_names is None:
        alphabets = 'abcdefghijklmnopqrstuvwxyz'
        try:
            gr_names = list(alphabets.upper() + alphabets)[:num_gr]
        except:
            raise ValueError('Please provide group names.')

    if model_names is None:
        model_names = [str(i) for i in range(num_model)]

    if model_markers is None:
        model_markers = ['${}$'.format(i) for i in model_names]
        model_markers[0] = '*'

    _colors = matplotlib.cm.get_cmap(gr_cmap)(np.linspace(0, 1, num_gr))

    gr_colors = _colors if gr_colors is None else gr_colors

    if len(model_names) != num_model or len(gr_names) != num_gr:
        raise ValueError(
            """Please check model_names and gr_names as their length is not
            in agreement with model_cc.

            Number of rows in model_cc should be equal to len(model_names) and
            Number of columns in model_cc should be equal to len(gr_names).
            """)
    if std_range is None:
        max_x = np.ceil(model_std.max())
        std_range = (0, max_x)

    dia = _draw_group_taylor(model_cc, model_std, model_names, gr_names, gr_colors, model_markers, model_markersize, std_range,*args, **kwargs)
    if show_gr_legend or show_model_legend:
        _add_legend_group_taylor(model_names, gr_names, gr_colors, model_markers, model_markersize, gr_desc, model_desc,show_gr_legend, show_model_legend,*args, **kwargs)

    return dia


def _draw_group_taylor(model_cc, model_std, model_names, gr_names, gr_colors, model_markers, model_markersize,std_range, *args, **kwargs):
    min_x, max_x = std_range

    for i in range(len(gr_names)):
        refstd = model_std[:,i].max()
        srange = (min_x/refstd, max_x/refstd)
        taylor_diag_options = dict(label=model_names[0],
                                   srange=srange,
                                   refstd_markercolor=gr_colors[i],
                                   refstd_marker=model_markers[0],
                                   refstd_linewidth=1,
                                   contours=True,
                                   refstd_markersize=model_markersize,
                                   contour_spec=dict(colors=[gr_colors[i]], linewidths=0.75,linestyles=':',alpha=1))
        if i==0:
            dia = TaylorDiagram(refstd,
                                *args,
                                **taylor_diag_options,
                                **kwargs)
        else:
            dia = TaylorDiagram(refstd,
                                *args,
                                ax=dia._ax,
                                **taylor_diag_options,
                                **kwargs)
    if kwargs.get('grid', []):
        dia.add_grid(True)

    for i in range(len(gr_names)):
        for j, (stddev, corrcoef) in enumerate(zip(model_std[1:,i], model_cc[1:,i])):
            dia.add_sample(stddev, corrcoef,
                           marker=model_markers[j+1], ms=model_markersize, ls='',
                           mfc=gr_colors[i], mec=gr_colors[i],
                           label=model_names[j+1])
    return dia

def _add_legend_group_taylor(model_names, gr_names, gr_colors, model_markers, model_markersize, gr_desc, model_desc,show_gr_legend,show_model_legend,*args, **kwargs):
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    fig = plt.gcf()
    if show_gr_legend:
        patch = []
        for c,g in zip(gr_colors, gr_names):
            patch.append(mpatches.Patch(color=c, label=g))

        fig.legend(handles=patch, title=gr_desc, loc='right')

    if show_model_legend:
        legend_elements = []
        for m,v in zip(model_markers, model_names):
            legend_elements.append(
                Line2D([0], [0], color='k', marker=m, markersize=model_markersize, label=v, lw=0))

        fig.legend(handles=legend_elements,
           numpoints=1,  loc='lower right', title=model_desc)
