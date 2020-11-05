#!/usr/bin/env bash
set -e

if ! type pyenv >/dev/null; then
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

    curl https://pyenv.run | bash

    echo 'export PATH="$HOME/.pyenv/bin:$HOME/.local/bin/:$PATH"' >>~/.bashrc
    echo 'eval "$(pyenv init -)"' >>~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >>~/.bashrc

    export PATH="$HOME/.pyenv/bin:$HOME/.local/bin/:$PATH"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    pyenv install 3.7.8
fi

pip install pipenv
python -m pipenv install
