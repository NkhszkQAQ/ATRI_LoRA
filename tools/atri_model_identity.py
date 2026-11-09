"""Compatibility import; implementation lives in the shared atri_core package."""
import sys
if __package__:
    from ._atri_core_bootstrap import core_module
else:
    from _atri_core_bootstrap import core_module

_impl = core_module("identity")
sys.modules[__name__] = _impl
