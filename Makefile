.PHONY: lint test

lint:
	poetry run ruff check src/ --fix --unsafe-fixes
	poetry run pyright src/

test:
	poetry run pytest src/tests/ -s
