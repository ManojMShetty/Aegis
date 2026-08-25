"""Tool implementations - and the guard that stands in front of the real ones.

There are no real tools in this package today. The guard is here first, on
purpose: see :mod:`aegis.tools.guard`.
"""

from aegis.tools.guard import RealToolImportError, require_real_tools_enabled

__all__ = ["RealToolImportError", "require_real_tools_enabled"]
