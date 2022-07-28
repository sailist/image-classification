"""

"""
import os
import sys

if os.environ.get('LUMO_LIB', None):
    sys.path.insert(0, os.environ.get('LUMO_LIB', None))

import importlib
from pprint import pprint
from track_semi.semitrainer import SemiParams
import track_semi
from pkgutil import iter_modules

methods = {i.name for i in list(iter_modules(track_semi.__path__))}

if __name__ == '__main__':
    params = SemiParams()
    params.module = None
    params.from_args()
    if params.module is None or params.module not in methods:
        print('--module=')
        pprint(methods)
        exit(1)
    module = importlib.import_module(f'track_semi.{params.module}')
    module.main()
