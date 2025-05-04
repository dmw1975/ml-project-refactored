"""DEPRECATED visualization module.

This package is deprecated and maintained only for backward compatibility.
Please use visualization_new package instead.

The original files have been moved to visualization_legacy.
"""

import warnings

warnings.warn(
    "The visualization module is deprecated. Please use visualization_new instead.",
    DeprecationWarning,
    stacklevel=2
)

# Redirect imports to visualization_new
from visualization_new import *
