.PHONY: clean format check install uninstall test pypi

venv:
	which python3
	python3 -m venv venv

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*pycache*" | xargs rm -rf

format:
	black reinforce
	blackdoc reinforce
	isort reinforce

check:
	black reinforce --check --diff
	blackdoc reinforce --check
	flake8 --config pyproject.toml --ignore E203,E501,W503 reinforce
	mypy --config pyproject.toml reinforce
	isort reinforce --check --diff

install:
	python3 setup.py install

uninstall:
	python3 -m pip uninstall reinforce -y

test:
	python3 -m pytest --doctest-modules
