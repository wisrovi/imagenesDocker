name := test_python_gpu
file_tf := test_gpu_tf.py
file_pt := test_gpu_pt.py


build-tf:
	sudo docker build -t $(name):tf -f Dockerfile.tf .

build-pt:
	sudo docker build -t $(name):pt -f Dockerfile.pt .

run-tf:
	sudo docker run --gpus all -it --rm -v $(PWD):/tmp -w /tmp $(name):tf python $(file_tf)

run-pt:
	sudo docker run --gpus all -it --rm -v $(PWD):/tmp -w /tmp $(name):pt python $(file_pt)