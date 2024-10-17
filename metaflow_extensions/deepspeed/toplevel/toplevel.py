__mf_extensions__ = "deepspeed"



try:
    import pkg_resources
    __version__ = pkg_resources.get_distribution("metaflow-deepspeed").version
except:
    # this happens on remote environments since the job package
    # does not have a version
    __version__ = None
