venv: 
	python -m venv venv
requirements:
	pip install -r requirements.txt
clean:
	black vae_path_generator
	black scripts
	isort vae_path_generator
	isort scripts
clear_logs:	
	rm -f logs/* 
