# opcion 1
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu    python -c "import tensorflow as tf; print(tf.__version__);print(len(tf.config.list_physical_devices('GPU')));"


# opcion 2
docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:latest-gpu python ./test_gpu.py
