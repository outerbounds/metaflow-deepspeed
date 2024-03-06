from metaflow.exception import MetaflowException


class DeepspeedException(MetaflowException):
    headline = "Deepspeed Command Failed"

    def __init__(self, msg):
        super(DeepspeedException, self).__init__(msg)


class DatastoreKeyNotFoundError(MetaflowException):
    headline = "DeepSpeed Datastore Not Found"

    def __init__(self, datastore_path_name):
        msg = "The DeepSpeed datastore path {} was not found.".format(
            datastore_path_name
        )
        super(DatastoreKeyNotFoundError, self).__init__(msg)


class BarrierTimeoutException(MetaflowException):
    headline = "Barrier Timeout"

    def __init__(self, lock_name, description):
        msg = f"Task has timed out after waiting for some keys to be written to the datastore.\n[Barrier Name]:{lock_name}\n[Barrier Info]: {description}"
        super(BarrierTimeoutException, self).__init__(msg)


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


class SSHScanUnsuccessfulException(MetaflowException):
    headline = "`ssh-keyscan` Unsuccessful"

    def __init__(
        self,
        bad_host,
        keyscan_error,
        ubf_context,
        node_index,
    ):
        _err = "[couldn't capture standard-error]"
        if keyscan_error is not None:
            _err = keyscan_error
        msg = f"Failed to capture the SSH keys using `ssh-keyscan` for the worker type {ubf_context} and node index {node_index}. `ssh-keyscan` failures indicate that the nodes are unable to communicate with each other via ssh. \n\n[Error]: {_err}\n\n[Bad Host]: {bad_host}"
        super(SSHScanUnsuccessfulException, self).__init__(msg)
