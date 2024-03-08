venv: 
	python -m venv venv
requirements:
	pip install -r requirements.txt
clean:
	black vaepi_sampler
	black scripts
	isort vaepi_sampler
	isort scripts
clear_logs:	
	rm -f logs/* 
