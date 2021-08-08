from pathlib import Path
import torch
import math
import glob
import logging

def make_divisible(x, divisor):
    # returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_file(file):
    """ search/download file (if necessary) and return path
    Arguments:
        file (str): file path
    """
    # make sure input is string
    file = str(file)
    if Path(file).is_file() or file == '':
        # if file is already existed, return the path
        return file
    elif file.startswith(("http://", "https://")):
        # if file is a link, download it
        url, file = file, Path(file).name
        print("Downloading {} to {}...".format(url, file))
        torch.hub.download_url_to_file(url, file)
        # check the downloaded file
        assert Path(file).exists() and Path(file).stat().st_size > 0, "File download failed: {}".format(url)
        return file
    else:
        # otherwise search the file
        files = glob.glob("./**/" + file, recursive = True)
        assert len(files), "File not found: {}".format(file)
        assert len(files) == 1, "Multiple files match '{}', specify exact path: {}".format(file, files)
        return files[0]

def set_logging(rank = -1, verbose = True):
    logging.basicConfig(format = "%(message)s", level = logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)
