import logging
import numpy as np
import pint
import sys
from netCDF4 import Dataset

from conversions import Q_, ureg

LOG = logging.getLogger(__name__)


def total_saturation_pressure(temp_in_k, mix_lo=253.16, mix_hi=273.16):
    """Calculate the total saturation pressure.

    :param temp_in_k: Temperature in kelvin at all pressure levels
    :param mix_lo:
    :param mix_hi:
    :return: Total saturation pressure
    """
    mix_lo = Q_(mix_lo, "K")
    mix_hi = Q_(mix_hi, "K")
    saturation_vapor_pressure_wmo = (vapor_pressure_liquid_water_wmo(temp_in_k))
    es_total = saturation_vapor_pressure_wmo.copy()

    vapor_pressure_ice = vapor_pressure_over_ice(temp_in_k)
    ice_ind = temp_in_k <= mix_lo
    es_total[ice_ind] = vapor_pressure_ice[ice_ind]

    mix_ind = (temp_in_k > mix_lo) & (temp_in_k < mix_hi)
    liq_weight = (temp_in_k - mix_lo) / (mix_hi - mix_lo)
    ice_weight = (mix_hi - temp_in_k) / (mix_hi - mix_lo)

    e_mix = ice_weight * vapor_pressure_ice + liq_weight * saturation_vapor_pressure_wmo
    es_total[mix_ind] = e_mix[mix_ind]

    return es_total


def vapor_pressure_liquid_water_wmo(temp_in_kelvin):
    """Calculate the Vapor pressure over liquid water below 0°C by WMO Formula."""
    # Saturation vapor pressure:
    #  http://faculty.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html
    # >273.16: w.r.t liquid water
    # 253.16 < T < 273.16: weighted interpolation of water / ice
    # < 253.16: w.r.t ice

    kelvinQuant = Q_(273.16, "K")

    es_wmo = 10.0 ** (10.79574 * (1. - kelvinQuant / temp_in_kelvin)
                      - 5.02800 * np.log10(temp_in_kelvin / kelvinQuant)
                      + 1.50475 * 10. ** -4. *
                      (1. - 10. ** (-8.2969 * (temp_in_kelvin / kelvinQuant - 1)))
                      + 0.42873 * 10. ** -3. *
                      (10. ** (4.76955 * (1 - kelvinQuant / temp_in_kelvin)) - 1.)
                      + 0.78614) * 100.0  # [Pa])

    return es_wmo


def vapor_pressure_over_ice(temp_in_kelvin):
    """Calculate the vapor pressure over ice using the Goff Gratch equation."""
    kelvinQuant = Q_(273.16, "K")
    goff_gratch_vapor_pressure_ice = (10.0 ** (
            -9.09718 * (kelvinQuant / temp_in_kelvin - 1.0)
            - 3.56654 * np.log10(kelvinQuant / temp_in_kelvin)
            + 0.876793 * (1.0 - temp_in_kelvin / kelvinQuant)
            + np.log10(6.1071)
    ) * 100.0)  # [Pa]

    return goff_gratch_vapor_pressure_ice


def vapor_pressure_approximation(qv, sfc_pressure, plevels):
    """Approximate the vapor pressure using specific humidity and surface pressure.

    :param qv: specific humidity of the water vapor
    :param sfc_pressure: surface pressure
    :param plevels: hPa levels of pressure
    :return: a good approximation of vapor pressure (still should multiply by pressure @ each level)
    """
    vapor_pressure = 1.0 / (
            0.622 / qv + (1.0 - 0.622)
    )  # still need to multiply by pressure @ each level
    if sfc_pressure is None:
        # 3D RH field
        for i, lev in enumerate(plevels.magnitude):
            # already cut out time dim
            vapor_pressure[i, :, :] = vapor_pressure[i, :, :] * lev
    else:
        # RH @ 10m: multiply by surface pressure
        vapor_pressure = vapor_pressure * sfc_pressure

    return vapor_pressure


def qv_to_rh(specific_humidity, temp_k, in_levels: pint.Quantity, press_at_sfc=None):
    """Convert Specific Humidity [kg/kg] -> relative humidity [%]."""
    # See Petty Atmos. Thermo. 4.41 (p. 65), 8.1 (p. 140), 8.18 (p. 147)
    levels = (in_levels.to(ureg.pascal))  # [hPa] -> [Pa] when necessary.

    # match fill values
    es_tot = total_saturation_pressure(temp_k)

    vapor_pressure = vapor_pressure_approximation(specific_humidity, press_at_sfc, levels)

    relative_humidity = (vapor_pressure / es_tot).magnitude * 100.0  # relative humidity [%]
    relative_humidity[relative_humidity > 100.0] = 100.0
    relative_humidity = relative_humidity.astype(np.float32)   # don't want double

    return relative_humidity


def rh_at_sigma(temp10m, sfc_pressure, sfc_pressure_fill, levels: pint.Quantity, data):
    """Calculate the rh at sigma using 10m fields."""
    temp_k = temp10m  # temperature in [K] (Y, X) not in (time, Y, X)

    # pressure in [Pa]
    sfc_pressure[sfc_pressure == sfc_pressure_fill] = np.nan

    rh_sigma = qv_to_rh(data, temp_k, levels, press_at_sfc=sfc_pressure)
    rh_sigma = rh_sigma.astype(np.float32)
    rh_sigma.set_fill_value = sfc_pressure_fill

    return rh_sigma


def dobson_layer(mmr_data_array, level_index, pressure_levels: pint.Quantity):
    """Calculate a dobson layer from a 3D mmr data array given a level index.

    :param pressure_levels: 1D pint Quantity array of pressure levels in hPa from data file.
    :param mmr_data_array: Mass mixing ratio data array
    :param level_index: index of current level
    :return: dobson layer
    """
    # Make Sure Pressure is in hPa
    pressure_levels = (pressure_levels.to(ureg.hectopascal)).magnitude

    dobson_unit_conversion = 2.69e16  # 1 DU = 2.69e16 molecules cm-2
    gravity = 9.8  # m/s^2
    avogadro_const = 6.02e23  # molecules/mol
    o3_molecular_weight = 0.048  # kg/mol
    dry_air_molecular_weight = 0.028966  # kg/mol molecular weight of dry air.

    const = 0.01 * avogadro_const / (gravity * dry_air_molecular_weight)
    vmr_j = mmr_data_array[level_index] * dry_air_molecular_weight / o3_molecular_weight
    vmr_j_1 = mmr_data_array[level_index - 1] * dry_air_molecular_weight / o3_molecular_weight
    ppmv = 0.5 * (vmr_j + vmr_j_1)
    delta_pressure = pressure_levels[level_index - 1] - pressure_levels[level_index]
    o3_dobs_layer = ppmv * delta_pressure * const / dobson_unit_conversion

    return o3_dobs_layer


def total_ozone(mmr, fill_value, pressure_levels: pint.Quantity):
    """Calculate total column ozone in dobson units (DU), from ozone mixing ratio in [kg/kg]."""
    nlevels = mmr.shape[0]
    dobson = np.zeros(mmr[0].shape).astype("float32")
    for j in range(1, nlevels):
        good_j = np.logical_not(np.isnan(mmr[j]))
        good_j_1 = np.logical_not(np.isnan(mmr[j - 1]))
        good = good_j & good_j_1
        dobs_layer = dobson_layer(mmr, j, pressure_levels)
        dobson[good] = dobson[good] + dobs_layer[good]

    dobson[np.isnan(dobson)] = fill_value

    return dobson


def merra_land_mask(data: np.ndarray, mask_sd: Dataset) -> np.ndarray:
    """Convert fractional merra land mask to 1=land 0=ocean.

    add FRLANDICE to include antarctica and greenland.
    :rtype: np.ndarray
    """
    frlandice = mask_sd.variables["FRLANDICE"][0]  # 0th time index
    data = frlandice + data
    return data > 0.25


def hack_snow(data: np.ndarray, mask_sd: Dataset) -> np.ndarray:
    """Force greenland/antarctica to be snowy like CFSR."""
    # Special case: set snow depth missing values to match CFSR behavior.
    frlandice = mask_sd.variables["FRLANDICE"][0]  # 0th time index
    data[frlandice > 0.25] = 100.0
    return data


# def apply_conversion(scale_func: Callable, data: np.ndarray, fill, p_levels=None) -> np.ndarray:
#     """Apply fill to converted data after function."""
#     converted = data.copy()
#
#     if scale_func == "total_ozone":
#         converted = total_ozone(data, fill, p_levels)
#     else:
#         converted = scale_func(converted)
#
#         if fill is not None:
#             converted[data == fill] = fill
#
#     return converted
