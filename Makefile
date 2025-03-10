SHELL := /bin/bash

.PHONY: install check format build

install:
	python3 -m pip install -r requirements.txt

check:
	ruff check *.py

format:	
	ruff check --fix
	ruff format

build:
	pip install -q build
	python3 -m build