"""
Canonical path locations for zipline data.

Paths are rooted at $ZIPLINE_ROOT if that environment variable is set.
Otherwise default to expanduser(~/.zipline)
"""
from errno import EEXIST
import os
from os.path import exists, expanduser, join

import pandas as pd


def ensure_directory(path):
    """
    Ensure that a directory named "path" exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == EEXIST and os.path.isdir(path):
            return
        raise


def ensure_directory_containing(path):
    """
    Ensure that the directory containing `path` exists.

    This is just a convenience wrapper for doing::

        ensure_directory(os.path.dirname(path))
    """
    ensure_directory(os.path.dirname(path))


def last_modified_time(path):
    """
    Get the last modified time of path as a Timestamp.
    """
    return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')


def modified_since(path, dt):
    """
    Check whether `path` was modified since `dt`.

    Returns False if path doesn't exist.

    Parameters
    ----------
    path : str
        Path to the file to be checked.
    dt : pd.Timestamp
        The date against which to compare last_modified_time(path).

    Returns
    -------
    was_modified : bool
        Will be ``False`` if path doesn't exists, or if its last modified date
        is earlier than or equal to `dt`
    """
    return exists(path) and last_modified_time(path) > dt


def zipline_root(environ=None):
    """
    Get the root directory for all zipline-managed files.
        if modified_since(path, now - ONE_HOUR):
            logger.warn(
                "Refusing to download new benchmark "
                "data because a download succeeded at %s." % (
                    last_modified_time(path)
                )
            )
            return data

    For testing purposes, this accepts a dictionary to interpret as the os
    environment.

    Parameters
    ----------
    environ : dict, optional
        A dict to interpret as the os environment.

    Returns
    -------
    root : string
        Path to the zipline root dir.
    """
    if environ is None:
        environ = os.environ.copy()

    root = environ.get('ZIPLINE_ROOT', None)
    if root is None:
        root = expanduser('~/.zipline')

    return root


def zipline_path(paths, environ=None):
    """
    Get a path relative to the zipline root.

    Parameters
    ----------
    paths : list[str]
        List of requested path pieces.
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    newpath : str
        The requested path joined with the zipline root.
    """
    return join(zipline_root(environ=environ), *paths)


def data_root(environ=None):
    """
    The root directory for zipline data files.

    Parameters
    ----------
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    data_root : str
       The zipline data root.
    """
    return zipline_path(['data'], environ=environ)


def ensure_data_root(environ=None):
    """
    Ensure that the data root exists.
    """
    ensure_directory(data_root(environ=environ))


def data_path(paths, environ=None):
    """
    Get a path relative to the zipline data directory.

    Parameters
    ----------
    paths : iterable[str]
        List of requested path pieces.
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    newpath : str
        The requested path joined with the zipline data root.
    """
    return zipline_path(['data'] + list(paths), environ=environ)


def cache_root(environ=None):
    """
    The root directory for zipline cache files.

    Parameters
    ----------
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    cache_root : str
       The zipline cache root.
    """
    return zipline_path(['cache'], environ=environ)


def ensure_cache_root(environ=None):
    """
    Ensure that the data root exists.
    """
    ensure_directory(cache_root(environ=environ))


def cache_path(paths, environ=None):
    """
    Get a path relative to the zipline cache directory.

    Parameters
    ----------
    paths : iterable[str]
        List of requested path pieces.
    environ : dict, optional
        An environment dict to forward to zipline_root.

    Returns
    -------
    newpath : str
        The requested path joined with the zipline cache root.
    """
    return zipline_path(['cache'] + list(paths), environ=environ)
