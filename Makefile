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
	poetry run black reinforce
	poetry run blackdoc reinforce
	poetry run isort reinforce

check:
	poetry run black reinforce --check --diff
	poetry run blackdoc reinforce --check
	poetry run flake8 --config pyproject.toml --ignore E203,E501,W503 reinforce
	poetry run mypy --config pyproject.toml reinforce --ignore-missing-imports
	poetry run isort reinforce --check --diff

install:
	poetry install
	poetry build

test:
	poetry run pytest --doctest-modules reinforce tests
