"""
Taylor diagram (Taylor, 2001) and Grouped Taylor Diagram implementation.

Reference:
Taylor, K.E., 2001. Summarizing multiple aspects of model performance in 
a single diagram. Journal of Geophysical Research: Atmospheres, 
106(D7), pp.7183-7192.

Suman, M., Maity, R. and Kunstmann, H., 2021. Precipitation of Mainland 
India: Copula‐based bias‐corrected daily CORDEX climate data for both mean 
and extreme values. Geoscience Data Journal.

Note: If you have found these software useful for your research, I would
appreciate an acknowledgment.
"""

__version__ = "v1.0"
__author__ = "Mayank Suman<mayanksuman@live.com> Taylor Diagram by Yannick Copin<yannick.copin@laposte.net>"

from .taylorDiagram import TaylorDiagram
from .group_taylor import group_taylor_diagram

__all__ = ["TaylorDiagram", "group_taylor_diagram"]
