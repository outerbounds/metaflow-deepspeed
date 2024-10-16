__mf_extensions__ = "deepspeed"


import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("metaflow-deepspeed").version
except:
    # this happens on remote environments since the job package
    # does not have a version
    __version__ = None
