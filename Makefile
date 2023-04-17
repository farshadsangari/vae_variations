rm_cache:
	find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" \) -exec rm -rf {} \;
dirs:
	mkdir -p data ckpts reports
requirements: dirs
	pip install -r requirements.txt