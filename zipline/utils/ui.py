"""
IPython-aware progress bar.
"""
from progressbar import (
    Bar,
    ETA,
    Percentage,
    ProgressBar,
)


def in_notebook():
    """
    Test whether we're running with an IPython Notebook frontend.

    This isn't really a supported API, so buyer beware here...
    """
    try:
        import IPython  # noqa
    except ImportError:
        return False
    try:
        cfg = get_ipython().config
        return 'connection_file' in cfg['IPKernelApp']
    except (NameError, KeyError):
        return False


def new_progress_bar(maxval):
    if in_notebook():
        return ProgressBar(
            widgets=[Percentage(), ' ', ETA(), Bar()],
            maxval=maxval
        )
    else:
        return ProgressBar(
            maxval=maxval
        )
