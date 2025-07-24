import os

if os.environ.get("LLUMUX_HOME") is None:
    os.environ["LLUMUX_HOME"] = ".."