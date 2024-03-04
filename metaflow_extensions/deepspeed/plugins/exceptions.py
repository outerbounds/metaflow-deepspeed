from metaflow.exception import MetaflowException


class DeepspeedException(MetaflowException):
    headline = ""

    def __init__(self, cmd):
        msg = "The Deepspeed command \n\n{}\n\nfailed to complete.".format(
            " ".join(cmd)
        )
        super(DeepspeedException, self).__init__(msg)


class DeepspeedDatastoreNotFoundError(MetaflowException):
    headline = "DeepSpeed Datastore Not Found"

    def __init__(self, datastore_path_name):
        msg = "The DeepSpeed datastore path {} was not found.".format(
            datastore_path_name
        )
        super(DeepspeedDatastoreNotFoundError, self).__init__(msg)
