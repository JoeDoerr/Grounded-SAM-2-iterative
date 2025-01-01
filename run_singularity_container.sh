#!/bin/bash

#docker save -o dockerimage.tar dockerimage:stuff
#scp dockerimage.tar username@server:wd
#singularity build output_image.sif docker-archive:/path/to/docker_image.tar
#For the cache I first copied the cached files that my docker image was using from its .cache to somewhere outside the running docker image
#scp -r .cache username@server:/common/home/jhd79/objects/.cache_gsam

#When you run the singularity image it has everything downloaded like a good venv, but the spawned filesystem can be weird
#Specifically with setting $HOME and thus calling .bashrc so we do that for it
#Also to cache the networks properly to load them fast we bind a cache from outside. Also binding to a modifiable repo makes this like an even better docker image.

singularity exec --nv --contain --no-home \
 --bind /common/home/jhd79/objects/.cache_gsam:/root/.cache,/common/home/jhd79/objects/Grounded-SAM-2-iterative:/home/appuser/Grounded-SAM-2 \
 ../gsam.sif /bin/bash --login -c "
export HOME=/root;
cd \$HOME;
source ./.bashrc;
echo 'Now gsam';
cd /home/appuser/Grounded-SAM-2;
python server_gsam.py
"