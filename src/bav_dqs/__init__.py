from importlib.metadata import version, PackageNotFoundError

try:
    # O nome aqui deve ser exatamente o "name" definido no [project] do seu pyproject.toml
    __version__ = version("bav-dqs-framework")
except PackageNotFoundError:
    # Caso o pacote não esteja instalado (ex: rodando localmente sem pip install -e .)
    __version__ = "unknown"