all: setup_environment

clean:
	poetry run pre-commit uninstall
	rm -rf .venv

setup_environment: check
		pyenv install 3.9 --skip-existing \
		&& pyenv local 3.9 \
		&& poetry env use 3.9 \
		&& poetry install \
		&& poetry run pre-commit install



create_folders:
	mkdir -p data/processed data/raw data/inter data/external notebooks models configs


check: pyenv_exists poetry_exists is_git

pyenv_exists: ; @which pyenv > /dev/null

poetry_exists: ; @which poetry > /dev/null

is_git: ; @git rev-parse --git-dir > /dev/null