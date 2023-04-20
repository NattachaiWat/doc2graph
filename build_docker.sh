docker image rm doc2graph:latest 
docker build -f dockerfile -t doc2graph:latest .
docker run --gpus all --shm-size=8g -it --rm -v $PWD:/code doc2graph:latest bash