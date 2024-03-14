"""Code source is based on Reichler et. al.

from:
http://www.inscc.utah.edu/~reichler/research/projects/TROPO/code.txt

   Reichler, T., M. Dameris, R. Sausen (2003):
   Determining the tropopause height from gridded data,
   Geophys. Res. L., 30, No. 20, 2042

Also conversion of some code from
http://www.cesm.ucar.edu/models/cesm1.2/cesm/cesmBbrowser/html_code/cam/tropopause.F90.html

History:
    9/2015 - Steve Wanzong - Created (fortran version)
    8/2022 - Joleen Feltz - Converted to Python
"""
import numpy as np
import xarray as xr


class LapsePressure:
    """Calculate midpoint pressure and other values for lapse rate pressure profile."""

    # Define constants.
    AVOGADRO_NUMBER = 6.02214e26  # Avogadro's number ~ molecules/k_mole
    BOLTZMANN = 1.38065e-23  # Boltzmann's constant ~ J/K/molecule
    GAS_CONST = AVOGADRO_NUMBER * BOLTZMANN  # Universal gas constant ~ J/K/k_mole

    MOLECULAR_WEIGHT_DRY_AIR = 28.966  # molecular weight dry air ~ kg/k_mole
    SPECIFIC_HEAT_DRY_AIR = 1.00464e3  # specific heat of dry air   ~ J/kg/K
    GRAVITY = 9.80616  # acceleration of gravity ~ m/s^2
    DRY_AIR_GAS_CONSTANT = GAS_CONST / MOLECULAR_WEIGHT_DRY_AIR  # Dry air gas constant ~ J/K/kg

    # R/Cp; gas constant for dry air
    CONST_KAP = (GAS_CONST / MOLECULAR_WEIGHT_DRY_AIR) / SPECIFIC_HEAT_DRY_AIR
    CONST_KA1 = CONST_KAP - 1.0
    CONSTANT_FACTOR = -GRAVITY / DRY_AIR_GAS_CONSTANT

    def __init__(
            self,
            pressure=None,) -> None:
        """Find Midpoint pressure profile from input list of pressure data."""
        self.pressures = pressure
        self.ckap_press = pressure ** self.CONST_KAP

        # pressure mean ^kappa
        self.pmk = self._midpoint_pressure_kappa()

        # pressure mean
        self.pm = self.pmk ** (1 / self.CONST_KAP)

    def _midpoint_pressure_kappa(self):
        """Calculate the midpoint pressure for each level."""
        pmk = np.full(self.pressures.shape, np.nan)
        p_kappa = self.pressures ** self.CONST_KAP
        for i in range(0, p_kappa.shape[0]-1):
            pmk[i] = 0.5 * (p_kappa[i] + p_kappa[i-1])
        pmk = xr.DataArray(pmk, dims=self.pressures.dims)

        return pmk

    def get_shape(self):
        """Return the shape of this pressure profile."""
        return self.pmk.shape


def point_a(t, ckap_press):
    """Get point_a of layer."""
    a = t.diff(dim="z", label="lower") / ckap_press.diff(dim="z", label="lower")
    return a


def get_tm(t, a, ckap_press, half_level_p):
    """Get Temperature at midpoint."""
    a_ckap_p = a * ckap_press.isel(z=slice(0, -1))
    b = t.sel(z=slice(0, -1)) - a_ckap_p
    tm = (a * half_level_p) + b

    return tm


def get_dtdp(a, pm_kappa, const_kap):
    """Get change in temperature over pressure."""
    a_mult_kappa = a * const_kap
    dtdp = a_mult_kappa * pm_kappa
    return dtdp


def possible_tropopause_pressures(dtdz_arr, pm, pmk, const_kap):
    """Find possible tropopause pressure levels in these profiles."""
    np.seterr(invalid="ignore")

    gamma = -0.002  # K / m
    upper_pressure_limit = 45000.0  # Pa
    lower_pressure_limit = 7500.0  # Pa

    dtdz0_matches = dtdz_arr.where((dtdz_arr <= gamma) | (pm < upper_pressure_limit))
    pmk0_matches = pmk.where((dtdz_arr <= gamma) | (pm < upper_pressure_limit))

    ag = dtdz0_matches.diff(dim="z", label="lower") / pmk0_matches.diff(dim="z", label="lower")
    bg = dtdz0_matches[0:-1] - (ag * pmk0_matches[0:-1])
    ptph_arr = np.exp(np.log(((gamma - bg) / ag)) / const_kap)

    # set where dtdz0 > gamma to midpoint_pressure
    ptph_arr = ptph_arr.where((dtdz_arr[0:-1] > gamma) |
                              (pm[0:-1] < upper_pressure_limit),
                              pm[0:-1])

    ptph_arr = ptph_arr.where((ptph_arr < upper_pressure_limit) &
                              (ptph_arr > lower_pressure_limit), np.nan)

    return ptph_arr


def final_tropopause_pressure(ptph_arr, dtdz2, pm_arr, p2km_arr, gamma=-0.002):
    """Cycle through candidate tropopause levels until found."""
    nz = ptph_arr.shape[0]
    nx = ptph_arr.shape[1]
    tropopause_pressure = xr.DataArray(np.full([nx], np.nan), dims="dim_0")

    for ii in range(nx):
        asum = 0.
        count = 0.
        for index in range(0, nz):
            ptph = ptph_arr[index][ii]
            if np.isnan(ptph) or ptph < 0:
                continue
            p2km = p2km_arr[index][ii]
            # test until apm < p2km
            index2 = index+1
            try:
                # pmk2 replaced by vector operation of LR.pm which is pmk2 ** (1/CONST_KAP)
                pm2 = pm_arr[index2][ii]
            except IndexError:
                break
            # Is there anywhere that pm2 is > tropopause pressure?
            if pm2 > ptph:
                continue  # should not happen (GOTO 110)
            if pm2 < p2km:
                tropopause_pressure[ii] = ptph
                break
            else:
                asum = asum + dtdz2[index2][ii]
                count = count + 1.
                aquer = asum/count

                if aquer < gamma:
                    break
    return tropopause_pressure


def reshape_to(inarray, just_like_arr):
    """Reshape inarray to the same shape and dimensions of just_like_arr."""
    nx = just_like_arr.shape[1]
    a_size = inarray.shape[0]
    out_array = np.tile(inarray.reshape(a_size, 1), nx)
    out_array = xr.DataArray(out_array, coords=just_like_arr.coords, dims=just_like_arr.dims)

    return out_array


def wmo_tropopause(temperature, LR):
    """Find tropopause pressure from input temperature profile.

    Description:
        Implementation of Reichler et al. [2003] in python.

     Example usage::
         tropo_press = wmo_tropopause(t, p)

     Args::
         Temperature profile at the grid point.
         Pressure Profile Class Object

     Returns::
         Tropopause pressure.
    """
    delta_z = 2000.0

    # set Lapse_Rate Variables for Every Temperature
    # for ii in range(temperature.sizes["dim_0"]):
    ckap_press = reshape_to(LR.ckap_press.data, temperature)

    pt_a = point_a(temperature, ckap_press)
    pm_arr = reshape_to(LR.pm.data[1:], pt_a)
    pmkappa = pm_arr ** LR.CONST_KA1
    half_level = reshape_to(LR.pmk.data[1:], pt_a)

    tm_arr = get_tm(temperature, pt_a, ckap_press, half_level)

    dtdp_arr = get_dtdp(pt_a, pmkappa, LR.CONST_KAP)
    dtdz_arr = (LR.CONSTANT_FACTOR * dtdp_arr) * (pm_arr / tm_arr)

    ptph_arr = possible_tropopause_pressures(dtdz_arr, pm_arr, half_level, LR.CONST_KAP)

    # calculate pressure at 2km above candidate tropopause pressures.
    p2km_arr = ptph_arr + delta_z * ((pm_arr.isel(z=slice(1, None)) /
                                      tm_arr.isel(z=slice(1, None))) *
                                     LR.CONSTANT_FACTOR)
    # p at ptph + 2km ??
    tropopause_pressure = xr.apply_ufunc(final_tropopause_pressure, ptph_arr,
                                         dtdz_arr.isel(z=slice(1, None)),
                                         pm_arr.isel(z=slice(1, None)),
                                         p2km_arr, input_core_dims=[["z", "dim_0"], ["z", "dim_0"],
                                                                    ["z", "dim_0"], ["z", "dim_0"]],
                                         output_core_dims=[["dim_0"]])

    return tropopause_pressure


def interpolate_missing(in_arr, interp_scheme="cubic"):
    """Interpolate over missing grid points."""
    from scipy.interpolate import griddata

    # Interpolate over missing values JMF CHECK THIS.
    data_mask = ~np.isnan(in_arr)

    x = np.arange(0, in_arr.shape[1])
    y = np.arange(0, in_arr.shape[0])
    # get only the valid values
    xx, yy = np.meshgrid(x, y)
    points = (xx[data_mask], yy[data_mask])
    values = in_arr[data_mask].ravel()

    interpolated_data = griddata(points, values, (xx, yy), method=interp_scheme)

    return interpolated_data


def ERA_TROPO_PRESSURE(pressure_profile, temperature_profile, array_order):
    """Calculate tropopause pressure from input temperature and pressure profile.

    Description:
          Calculate the estimated tropopause pressure from a
          2D ERA5 temperature profile (height, elements) at a grid point.

    Example Usage::
           calc_tropopause_press = ERA_TROPO_PRESSURE(t, p, array_order)

    Args::
           P: 1D pressure level array.
           T: 3D temperature array.
           array_order:  XYZ or ZXY

    Returns::
        Calculated tropopause pressure array, 2D.
        Calculated tropopause temperature array, 2D.

    Dependencies:
        wmo_tropopause method which does the majority of the tropopause calculation.
        interpolate_missing method to fill in failed calculations with the mean of the
          surrounding grid points.

    History:
        09-2015 - Steve Wanzong - Created in fortran.
        06-2022 - Joleen Feltz  - Converted to python.
    """
    # convert pressure array from hPa to Pa
    pa = pressure_profile * 100.0

    # Flip pressure and temperature from ground to ionosphere.
    flip = False
    if pa[0] < pa[1]:
        pa = np.flip(pa, axis=0)
        flip = True
    pa = xr.DataArray(pa, dims=["z"])

    if array_order == "XYZ":
        t_profile = xr.DataArray(temperature_profile, dims=["x", "y", "z"])
        t_profile = t_profile.transpose("z", "x", "y")
    else:
        t_profile = xr.DataArray(temperature_profile, dims=["z", "x", "y"])

    if flip:
        # Need to flip the Temperature Profile Along Z axis.
        t_profile = t_profile.sel(z=slice(None, None, -1))

    PressureProfile = LapsePressure(pa)

    nx = t_profile.sizes["x"]
    ny = t_profile.sizes["y"]

    t_stacked = t_profile.stack(dim_0=('x', 'y'))

    tropopause_p = wmo_tropopause(t_stacked, PressureProfile)

    tropopause_p = tropopause_p.data.reshape(nx, ny)
    tropopause_p = interpolate_missing(tropopause_p)

    import matplotlib.pyplot as plt
    plt.imshow(tropopause_p)

    return tropopause_p


def ERA_TROPO_TEMPERATURE(tropopause_field, pressure_profile, temperature_profile):
    """Calculate tropopause temperature from input temperature and pressure profile.

    Description:
          Calculate the estimated tropopause temperature field from a
          3D ERA5 temperature profile (lat, lon, pressure).
    Args::
           tropopause_field: 2D calculated tropopause pressure
           pressure_profile: 1D pressure profile "z" of temperature_profile.
           temperature_profile:  3D temperature field (lat, lon, pressure)

    Returns::
        Calculated tropopause temperature array, 2D.

    Dependencies:
        wmo_tropopause method to calculate tropopause pressure if not already in dataset.
    """
    nx = temperature_profile.sizes["x"]
    ny = temperature_profile.sizes["y"]

    # Allocate the tropopause level array.
    tropopause_temperature = np.full([nx, ny], -999.)

    # Choose the tropopause level
    temporary_pressure = tropopause_field / 100.0

    # make sure input pressure profile is increasing.
    if np.diff(pressure_profile) < 0:
        pressure_profile = np.flip(pressure_profile)

    # stack the tropopause field
    temporary_pressure_stacked = (temporary_pressure.stack(dim_0=('x', 'y'))).data

    # Interpolate Tropopause Temperature
    dp = pressure_profile.diff(dim="z", label="lower")
    dt = temperature_profile.diff(dim="z", label="lower")
    temperature_stacked = (temperature_profile.stack(dim_0=('x', 'y'))).data

    for i in range(nx*ny):
        # first location in increasing pressure profile where this grid point
        # is greater than pressure at that level
        index_z = np.where(temporary_pressure_stacked[i] > pressure_profile)
        (x, y) = np.unravel_index(i, (nx, ny))
        temporary_pressure[x][y] = index_z

        if dp[index_z] != 0.0:
            tropopause_temperature[x][y] = temperature_stacked[index_z][i] + \
                                           (dt[index_z][i] / dp[index_z][i]) * \
                                           (temporary_pressure_stacked[i] -
                                            pressure_profile[index_z])
        else:
            tropopause_temperature[x][y] = temperature_profile[index_z][i]

    return tropopause_temperature
