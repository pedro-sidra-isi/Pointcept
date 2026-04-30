#!/bin/sh
# vim:sw=4:ts=4:et
set -e

#apt-get install -y fuse nodejs npm
#npm install -g n
#npm install -g npm@latest
#n 14.0 && hash -r
git config --global --add safe.directory /workspace
python -m pip install -e .
python -m pip install -e ./Pointcept
python -m pip install perlin-numpy
#pip install black
# nvim --headless -c "quit"
# rm /root/.local/share/nvim/mason/staging/pyright/package.json
#nvim -c "LspInstall ruff_lsp"
# nvim --headless -c "TSInstall python" -c "quit"

exec "$@"
