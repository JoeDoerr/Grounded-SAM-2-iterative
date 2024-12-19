#! /usr/bin/env bash
docker build -t groundedsam .
docker run -it --gpus all groundedsam bash
sleep 5
python huggingface_downloads.py
exit
echo now do docker commit <container_id> groundedsam .
