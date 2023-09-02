########## Pull ##########
FROM nvidia/cuda:11.6.2-base-ubuntu20.04
########## User ##########
ARG home_dir="/home/user"
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN apt-get update && \
	apt-get install -y \
		gosu \
		sudo && \
	chmod +x /usr/local/bin/entrypoint.sh && \
	mkdir -p $home_dir
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
########## Non-interactive ##########
ENV DEBIAN_FRONTEND=noninteractive
########## Common Tools ##########
RUN apt-get update && \
    apt-get install -y \
	    vim \
    	wget \
    	unzip \
    	git \
		python3-tk
########## PyTorch ##########
RUN apt-get update && \
    apt-get install -y \
	    python3-pip && \
	pip3 install \
		torch==1.13.1+cu116 \
		torchvision==0.14.1+cu116 \
		torchaudio==0.13.1 \
		--extra-index-url https://download.pytorch.org/whl/cu116 \
		tensorboardX \
		tensorboard
########## OpenAI ##########
RUN apt-get update && \
    apt-get install -y \
		libgl1-mesa-dev \
		python-opengl && \
	pip3 install \
		gymnasium[all] \
		stable-baselines3[extra] \
		sb3-contrib
########## Mujoco ##########
RUN apt-get update && \
    apt-get install -y \
		libosmesa6-dev \
		patchelf && \
	cd /tmp && \
	wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
	tar xvzf mujoco210-linux-x86_64.tar.gz && \
	mkdir $home_dir/.mujoco && \
	mv mujoco210 $home_dir/.mujoco/ && \
	chmod -R 777 /usr/local/lib/python3.8/dist-packages/mujoco_py
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$home_dir/.mujoco/mujoco210/bin
########## Initial Position ##########
WORKDIR $home_dir
CMD ["bash"]