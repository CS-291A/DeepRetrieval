import os


def load_secret(env_var, key_file):
    """Return the credential from `env_var` if set, else the stripped contents of `key_file`.

    `key_file` is resolved relative to the current working directory, as the original
    import-time reads were. Raises RuntimeError naming both sources if neither has a value.
    """
    value = os.environ.get(env_var, "").strip()
    if value:
        return value
    if os.path.isfile(key_file):
        with open(key_file, 'r') as f:
            value = f.read().strip()
        if value:
            return value
    raise RuntimeError(
        f"No credential found: set ${env_var} or create {os.path.abspath(key_file)}"
    )
