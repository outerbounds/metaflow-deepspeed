FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenSSH, OpenMPI, and some debugging utilities
RUN apt-get update -y && apt-get upgrade -y && apt-get install sudo vim wget net-tools openssh-server openssh-client gcc g++ openmpi-bin openmpi-common pdsh libopenmpi-dev libgtk2.0-dev libaio-dev python-is-python3 python3-pip -y && apt-get clean && apt-get purge

# Install Deepspeed and friends
RUN /usr/bin/python -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN /usr/bin/python -m pip install --user ninja transformers datasets accelerate pytorch-lightning pandas scikit-learn mpi4py
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 /usr/bin/python -m pip install --user deepspeed --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1
ENV PATH="${PATH}:/root/.local/bin"

# Build OpenMPI
RUN wget https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.1.tar.gz && tar -xvzf openmpi-4.1.1.tar.gz
RUN cd openmpi-4.1.1 && ./configure --prefix=/usr/local && make && make install
ENV PATH=”$PATH:/home/.openmpi/bin”
ENV LD_LIBRARY_PATH=”$LD_LIBRARY_PATH:/home/.openmpi/lib”

# Start SSH service
RUN service ssh start

# Expose docker port 22
EXPOSE 22

# Start SSH daemon - Need to do in Metaflow, since Metaflow replaces CMD
CMD ["/usr/sbin/sshd","-D"]