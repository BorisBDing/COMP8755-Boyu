import inspect
import os
import sys
from executors.clip_torch import *

if __name__ == '__main__':

    from jina import Flow

    if len(sys.argv) > 1:
        if sys.argv[1] == '-i':
            _input = sys.stdin.read()
        else:
            _input = sys.argv[1]
    else:
        _input = 'torch-flow.yml'

    f = Flow.load_config(
        _input,
        extra_search_paths=[os.path.dirname(inspect.getfile(inspect.currentframe()))],
    )
    with f:
        f.block()
