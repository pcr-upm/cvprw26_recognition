#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --ssh default=$HOME/.ssh/id_rsa -t cvprw26_recognition_image .
sudo docker run --name cvprw26_recognition_container --rm --gpus all -it -d cvprw26_recognition_image bash
sudo docker exec -w /home/username/cvprw26_recognition cvprw26_recognition_container python test/cvprw26_recognition_test.py --input-data test/example.jpg --database affectnet --gpu 0 --save-image
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo docker cp cvprw26_recognition_container:/home/username/conda/envs/cvprw26/lib/python3.12/site-packages/images_framework/output/images/. output/
sudo chown -R "${USER}":"${USER}" output
sudo docker rm -f cvprw26_recognition_container
sudo docker image rm cvprw26_recognition_image
sudo docker builder prune -a -f