import json
import os
import sys
import threading
import torch
from packaging.version import Version
from urllib.request import Request, urlopen

import pkg_resources
from rich import print
from rich.panel import Panel

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get('clip_server').__file__
        if 'clip_server' in sys.modules
        else __file__
    ),
    'resources',
)


__cast_dtype__ = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}

