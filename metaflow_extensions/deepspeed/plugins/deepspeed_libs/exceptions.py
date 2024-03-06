from metaflow.exception import MetaflowException


class DeepspeedException(MetaflowException):
    headline = ""

    def __init__(self, cmd):
        msg = "The Deepspeed command \n\n{}\n\nfailed to complete.".format(
            " ".join(cmd)
        )
        super(DeepspeedException, self).__init__(msg)


class DatastoreKeyNotFoundError(MetaflowException):
    headline = "DeepSpeed Datastore Not Found"

    def __init__(self, datastore_path_name):
        msg = "The DeepSpeed datastore path {} was not found.".format(
            datastore_path_name
        )
        super(DatastoreKeyNotFoundError, self).__init__(msg)


class WaitLockFailed(MetaflowException):
    headline = "Wait Lock Failed"

    def __init__(self, lock_name, description):
        msg = f"A task has timedout after waiting for some keys to be written to the datastore.\n[Lock Name]:{lock_name}\n[Lock Info]: {description}"
        super(WaitLockFailed, self).__init__(msg)


class SSHKeyGenException(MetaflowException):
    headline = "SSH Key Generation Failed"

    def __init__(self, cmd, exception, worker_type, node_index):
        msg = f"Failed to generate SSH keys for the worker type {worker_type} and node index {node_index}. \n\n[Command]: {cmd}\n\n[Exception]: {exception}"
        super(SSHKeyGenException, self).__init__(msg)


class SSHServiceRestartException(MetaflowException):
    headline = "SSH Service Restart Failed"

    def __init__(self, cmd, exception, worker_type, node_index):
        msg = f"Failed to restart the SSH service for the worker type {worker_type} and node index {node_index}. \n\n[Command]: {cmd}\n\n[Exception]: {exception}"
        super(SSHServiceRestartException, self).__init__(msg)
