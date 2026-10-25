try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

try:
    import pandas as pd
except ImportError:
    pd = None
else:
    pv = pd.__version__
