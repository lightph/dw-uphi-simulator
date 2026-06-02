from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['fourier_batch']
def fourier_batch(dc_bias: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], restoring_amplitude: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], forcing_amplitude: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], timestep: typing.SupportsFloat, wait_n_periods: typing.SupportsInt, average_n_periods: typing.SupportsInt, t0: typing.SupportsFloat, tf: typing.SupportsFloat = 0.0, x0: typing.SupportsFloat = 1.0) -> list:
    """
    Compute a batch of Fourier transforms
    """
