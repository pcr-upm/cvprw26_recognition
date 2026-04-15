#!/bin/bash
echo 'Using Docker to start the container and run tests ...'
sudo docker build --force-rm --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" -t cvprw26_recognition_image .
sudo docker volume create --name cvprw26_recognition_volume
sudo docker run --name cvprw26_recognition_container -v cvprw26_recognition_volume:/home/username --rm --gpus all -it -d cvprw26_recognition_image bash
sudo docker exec -w /home/username/cvprw26_recognition cvprw26_recognition_container python test/cvprw26_recognition_test.py --input-data test/example.tif --database affwild2 --gpu 0 --save-image
sudo docker stop cvprw26_recognition_container
echo 'Transferring data from docker container to your local machine ...'
mkdir -p output
sudo chown -R "${USER}":"${USER}" /var/lib/docker/
rsync --delete -azvv /var/lib/docker/volumes/cvprw26_recognition_volume/_data/conda/envs/cvprw26/lib/python3.12/site-packages/images_framework/output/images/ output
sudo docker system prune --all --force --volumes
sudo docker volume rm $(sudo docker volume ls -qf dangling=true)
