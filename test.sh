pytest -s
black -l 80 . --check
isort -rc . --check-only