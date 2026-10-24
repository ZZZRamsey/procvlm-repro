# Runtime-only compatibility shim; no environment files are changed.
import importlib.metadata as _metadata
_orig_version = _metadata.version
def _version(name):
    if name.lower() in {"huggingface-hub", "huggingface_hub"}:
        return "0.34.0"
    return _orig_version(name)
_metadata.version = _version
