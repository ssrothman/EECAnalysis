from __future__ import annotations

from hist import Hist
from coffea.processor.accumulator import AccumulatorABC

from hist.axestuple import NamedAxesTuple
from hist.axis import AxisProtocol
from hist.quick_construct import MetaConstructor
from hist.storage import Storage
from hist.svgplots import html_hist, svg_hist_1d, svg_hist_1d_c, svg_hist_2d
from hist.typing import ArrayLike, Protocol, SupportsIndex

from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple, TypeVar, Union
import typing 

import numpy as np

if typing.TYPE_CHECKING:
    from builtins import ellipsis

    import matplotlib.axes
    from mplhep.plot import Hist1DArtists, Hist2DArtists

    from hist.plot import FitResultArtists, MainAxisArtists, RatiolikeArtists


import functools
import operator
import typing
import warnings

class Histogram(Hist, family=None):
    def __init__(
        self,
        *args: AxisProtocol | Storage | str | tuple[int, float, float],
        storage: Storage | str | None = None,
        metadata: Any = None,
        data: np.typing.NDArray[Any] | None = None,
        label: str | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(*args)

    def identity(self):
        return self.copy().reset()