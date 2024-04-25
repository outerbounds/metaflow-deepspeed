__mf_extensions__ = "deepspeed"


import pkg_resources
from ..plugins.deepspeed_libs.hugging_face import card_callback as huggingface_card_callback

try:
    __version__ = pkg_resources.get_distribution("metaflow-deepspeed").version
except:
    # this happens on remote environments since the job package
    # does not have a version
    __version__ = None
