"""Game character painting plugin.

The package entrypoint intentionally contains no command logic. Importing each
handler module registers its matchers with NoneBot.
"""

from .handlers import alias_management as _alias_management  # noqa: F401
from .handlers import collection as _collection  # noqa: F401
from .handlers import draw as _draw  # noqa: F401
from .handlers import feature_filters as _feature_filters  # noqa: F401
from .handlers import feature_query as _feature_query  # noqa: F401
from .handlers import help as _help  # noqa: F401
from .handlers import lifecycle as _lifecycle  # noqa: F401
from .handlers import package_painting as _package_painting  # noqa: F401
from .handlers import rename as _rename  # noqa: F401
from .handlers import statistics as _statistics  # noqa: F401
from .metadata import __plugin_meta__

__all__ = ["__plugin_meta__"]
