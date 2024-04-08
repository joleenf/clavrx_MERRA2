"""Unit conversions for variables used in merra2 and era5 data."""
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

CLAVRX_FILL = 9.999e20
COMPRESSION_LEVEL = 6


def geopotential(data, make_quantity=True):
    """Convert geopotential in meters per second squared to geopotential height."""
    # this is height/1000.0*g
    if make_quantity:
        gravity = Q_(9.8, "m/s^2")
    else:
        gravity = 9.8

    return data / (1000 * gravity)
