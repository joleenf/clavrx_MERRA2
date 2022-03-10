""" TODO module doc """
from glob import glob
from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np
np.seterr(all='ignore')
import os
import sys

comp_level = 6  # 6 is the gzip default; 9 is best/slowest/smallest file

no_conversion = lambda a: a  # ugh why doesn't python have a no-op function...
fill_bad = lambda a: a*np.nan

# this is trimmed to the top CFSR level (i.e., exclude higher than 10hPa)
#LEVELS = [ 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
#    650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40,
#        30, 20, 10 ] # [hPa]
LEVELS = [ 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 40,
        30, 20, 10, 7, 5, 4, 3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.1 ] # [hPa]


def qv_to_rh(qv, t, ps=None):
    """ Specific Humidity [kg/kg] -> relative humidity [%] """
    # See Petty Atmos. Thermo. 4.41 (p. 65), 8.1 (p. 140), 8.18 (p. 147)
    levels = map(lambda a: a * 100.0, LEVELS) # [hPa] -> [Pa]

    # Saturation vapor pressure:
    #  http://faculty.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html
    # >273.16: w.r.t liquid water
    # 253.16 < T < 273.16: weighted interpolation of water / ice
    # < 253.16: w.r.t ice
    mix_lo = 253.16
    mix_hi = 273.16
    mix_ind = (t > mix_lo) & (t < mix_hi)
    ice_ind = t <= mix_lo
    es_wmo = 10.0**(10.79574*(1.-273.16/t) 
            - 5.02800*np.log10(t/273.16)
            + 1.50475*10.**-4.*(1.-10.**(-8.2969*(t/273.16-1)))
            + 0.42873*10.**-3.*(10.**(4.76955*(1-273.16/t)) - 1.)
            + 0.78614) * 100.0   # [Pa]
    es_tot = es_wmo.copy()
    ei_gg = 10.0 ** (
                -9.09718*(273.16/t - 1.)
                - 3.56654*np.log10(273.16/t)
                + 0.876793*(1. - t/273.16)
                + np.log10(6.1071)
            ) * 100.0 # [Pa]
    es_tot[ice_ind] = ei_gg[ice_ind]
    liq_weight = (t - mix_lo) / (mix_hi - mix_lo)
    ice_weight = (mix_hi - t) / (mix_hi - mix_lo)
    emix = ice_weight*ei_gg + liq_weight*es_wmo
    es_tot[mix_ind] = emix[mix_ind]

    # Vapor pressure e, to "a good approximation":
    #e = qv / 0.622 # still need to multiply by pressure @ each level
    # or, using unapproximated form:
    e = 1.0 / ( 0.622/qv + (1.0 - 0.622) ) # still need to multiply by pressure @ each level
    if ps is None:
        # 3D RH field
        for i, lev in enumerate(levels):
            e[i, :, :] = e[i, :, :] * lev # we've already cut out time dim
    else:
        # RH @ 10m: multiply by surface pressure
        e = e * ps
    rh = e / es_tot * 100.0 # relative humidity [%]
    rh[rh > 100.0] = 100.0 # clamp to 100% to mimic CFSR
    return rh

def kgkg_to_dobson(data):
    """Convert Ozone mixing ratio in [kg/kg] to dobson units."""
    du = 2.69e16  # TODO add units+doc from Mike's spreadsheet.
    g = 9.8
    av = 6.02e23
    mq = 0.048
    md = 0.028966
    const = 0.01 * av / (g * md)
    nlevels = data.shape[0]
    total = np.zeros(data[0].shape).astype('float32')
    for j in range(1, nlevels):
        mmr_j = data[j]
        mmr_j_1 = data[j-1]
        good_j = np.logical_not(np.isnan(mmr_j))
        good_j_1 = np.logical_not(np.isnan(mmr_j_1))
        good = good_j & good_j_1
        vmr_j = mmr_j * md / mq
        vmr_j_1 = mmr_j_1 * md / mq
        ppmv = 0.5*(vmr_j + vmr_j_1)
        dp = LEVELS[j-1] - LEVELS[j]
        dobs_layer = ppmv * dp * const / du
        total[good] = (total[good] + dobs_layer[good])
    return total 

# the merra4clavrx rosetta stone
# one key for each output var
rs = {
    # --- data vars from 'inst6_3d_(ana)_Np'
    'MSL pressure': {
        'in_file': 'ana',
        'in_varname': 'SLP',
        'out_units': 'hPa',
        'units_fn': lambda a: a/100.0, # scale factor for Pa --> hPa
        'ndims_out': 2
        },
    'temperature': {
        'in_file': 'ana',
        'in_varname': 'T',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 3
        },
    'surface pressure': {
        'in_file': 'ana',
        'in_varname': 'PS',
        'out_units': 'hPa',
        'units_fn': lambda a: a/100.0, # scale factor for Pa --> hPa
        'ndims_out': 2
        },
    'height': { 
        'in_file': 'ana',
        'in_varname': 'H',
        'out_units': 'km',
        'units_fn': lambda a: a/1000.0, # scale factor for m --> km
        'ndims_out': 3
        },
    'u-wind': {
        'in_file': 'ana',
        'in_varname': 'U',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 3
        },
    'v-wind': {
        'in_file': 'ana',
        'in_varname': 'V',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 3
        },
    'rh': {
        'in_file': 'ana',
        'in_varname': 'QV',
        'out_units': '%',
        'units_fn': None, # special case due to add'l inputs
        'ndims_out': 3
        },
    'total ozone': {
        'in_file': 'ana',
        'in_varname': 'O3',
        'out_units': 'Dobson',
        'units_fn': None, # special case due to add'l inputs
        'ndims_out': 2
        },
    'o3mr': {
        'in_file': 'ana',
        'in_varname': 'O3',
        'out_units': 'kg/kg',
        'units_fn': no_conversion,
        'ndims_out': 3
        },
    # --- data vars from 'tavg1_2d_(slv)_Nx'
    'tropopause pressure': {
        'in_file': 'slv',
        'in_varname': 'TROPPT',
        'out_units': 'hPa',
        'units_fn': lambda a: a/100.0, # scale factor for Pa --> hPa
        'ndims_out': 2
        },
    'tropopause temperature': {
        'in_file': 'slv',
        'in_varname': 'TROPT',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    'u-wind at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'U10M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    'v-wind at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'V10M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    'surface temperature': { # XXX confirm skin temp is correct choice for 'surface temperature'
        'in_file': 'slv',
        'in_varname': 'TS',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    'temperature at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'T10M',
        'out_units': 'K',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    'rh at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'QV10M',
        'out_units': '%',
        'units_fn': fill_bad, # XXX how to get p at sigma=0.995 for RH conversion?
        'ndims_out': 2
        },
    'u-wind at 50M': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'U50M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    'v-wind at 50M': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'V50M',
        'out_units': 'm/s',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    # --- data vars from 'tavg1_2d_(flx)_Nx'
    'planetary boundary layer height': {
        'in_file': 'flx',
        'in_varname': 'PBLH',
        'out_units': 'km',
        'units_fn': lambda a: a/1000.0, # scale factor for m --> km
        'ndims_out': 2
        },
    'ice fraction': {
        'in_file': 'flx',
        'in_varname': 'FRSEAICE',
        'out_units': 'none',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    # --- data vars from 'tavg1_2d_(lnd)_Nx'
    'water equivalent snow depth': {
        'in_file': 'lnd',
        'in_varname': 'SNOMAS',
        'out_units': 'kg/m^2',
        'units_fn': no_conversion, # special case in do_conversion will set fill values to zero
        'ndims_out': 2
        },
    # --- data vars from 'inst3_3d_(asm)_Np'
    'clwmr': {
        'in_file': 'asm3d',
        'in_varname': 'QL',
        'out_units': 'kg/kg',
        'units_fn': no_conversion,
        'ndims_out': 3
        },
    'cloud ice water mixing ratio': {
        'in_file': 'asm3d',
        'in_varname': 'QI',
        'out_units': 'kg/kg',
        'units_fn': no_conversion,
        'ndims_out': 3
        },
    # --- data vars from 'inst1_2d_(asm)_Nx'
    'total precipitable water': {
        'in_file': 'asm2d',
        'in_varname': 'TQV',
        'out_units': 'cm',
        'units_fn': lambda a: a/10.0, # scale factor for kg/m^2 (mm) --> cm
        'ndims_out': 2
        },
    # --- data vars from 'tavg1_2d_(rad)_Nx'
    'total cloud fraction': {
        'in_file': 'rad',
        'in_varname': 'CLDTOT',
        'out_units': 'none',
        'units_fn': no_conversion,
        'ndims_out': 2
        },
    # --- geoloc vars from 'ana'
    'lon': {
        'in_file': 'ana',
        'in_varname': 'lon',
        'out_units': None,
        'units_fn': no_conversion,
        'ndims_out': 1
        },
    'lat': {
        'in_file': 'ana',
        'in_varname': 'lat',
        'out_units': None,
        'units_fn': no_conversion,
        'ndims_out': 1
        },
    'pressure levels': {
        'in_file': 'ana',
        'in_varname': 'lev',
        'out_units': 'hPa',
        'units_fn': no_conversion, # yes this is indeed correct.
        'ndims_out': 1
        },
    }

def nc4_to_SD_dtype(nc4_dtype):
    # netCDF4 stores dtype as a string, pyhdf.SD stores dtype as a symbolic
    # constant. To properly convert, we need to go through an if-trap series
    # to identify the appropriate SD_dtype
    #
    # SD_dtype = nc4_to_SD_dtype(nc4_dtype)
    #
    # see, e.g. https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int32
    # to troubleshoot when an unassigned nc4_dtype appears
    #
    SD_dtype = None
    if (nc4_dtype == 'single') | (nc4_dtype == 'float32'):
        SD_dtype = SDC.FLOAT32
    elif (nc4_dtype == 'double') | (nc4_dtype == 'float64'):
        SD_dtype = SDC.FLOAT64
    elif nc4_dtype == 'uint32':
        SD_dtype = SDC.UINT32
    elif nc4_dtype == 'int32':
        SD_dtype = SDC.INT32
    elif nc4_dtype == 'uint16':
        SD_dtype = SDC.UINT16
    elif nc4_dtype == 'int16':
        SD_dtype = SDC.INT16
    elif nc4_dtype == 'int8':
        SD_dtype = SDC.INT8
    elif nc4_dtype == 'char':
        SD_dtype = SDC.CHAR
    else:
        raise ValueError("UNSUPPORTED NC4 DTYPE FOUND:",nc4_dtype)
    return SD_dtype





def _reshape(data, ndims_out, fill):
    """ Do a bunch of manipulation needed to make MERRA look like CFSR:
    
      * For arrays with dims (level, lat, lon) we need to make level the last dim.
      * All CFSR fields are continuous but MERRA sets below-ground values to fill.
      * CFSR starts at 0 deg lon but merra starts at -180.
    """
    if (ndims_out == 3) or (ndims_out == 2):
        data = _shift_lon(data)
    if ndims_out != 3:
        return data
    # do extrapolation before reshape
    # (extrapolate fn depends on a certain dimensionality/ordering)
    data = _extrapolate_below_sfc(data, fill)
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    data = data[:,:,::-1] # clavr-x needs toa->surface not surface->toa
    return data

# The highest (closest to TOA) level in CFSR. We trim anything above this:
TOP_LEVEL = 10 # [hPa]
CLAVRX_FILL = 9.999e20
def _refill(data, old_fill):
    """Clavr-x assumes a particular fill value instead of reading from attributes"""
    if (data.dtype == np.float32) or (data.dtype == np.float64):
        data[np.isnan(data)] = CLAVRX_FILL
        data[data == old_fill] = CLAVRX_FILL
    return data

def _shift_lon_2d(data):
    """ Helper function that assumes dims are 2d and (lat, lon) """
    nlon = data.shape[1]
    halfway = nlon // 2
    tmp = data.copy()
    data[:, 0:halfway] = tmp[:, halfway:]
    data[:, halfway:] = tmp[:, 0:halfway]
    return data

def _shift_lon(data):
    """ Make lon start at 0deg instead of -180. 
    
    Assume dims are (level, lat, lon) or (lat, lon)
    """
    if len(data.shape) == 3:
        for l_ind in np.arange(data.shape[0]):
            data[l_ind] = _shift_lon_2d(data[l_ind])
    elif len(data.shape) == 2:
        data = _shift_lon_2d(data)
    return data

def _extrapolate_below_sfc(t, fill):
    """Major difference between CFSR and MERRA is that CFSR extrapolates
       below ground and MERRA sets to fill. For now, we just try setting below
       ground fill values to lowest good value instead of fancy exptrapolation.
    """
    # Algorithm: For each pair of horizontal indices, find the lowest vertical index
    #            that is not CLAVRX_FILL. Use this data value to fill in missing
    #            values all the down to bottom index.
    lowest_good = t[0] * 0.0 + fill
    lowest_good_ind = np.zeros(lowest_good.shape, dtype=np.int)
    for l_ind in np.arange(t.shape[0]):
        t_at_l = t[l_ind]
        t_at_l_good = (t_at_l != fill)
        replace = t_at_l_good & (lowest_good == fill)
        lowest_good[replace] = t_at_l[replace]
        lowest_good_ind[replace] = l_ind

    for l_ind in np.arange(t.shape[0]):
        t_at_l = t[l_ind]
        # Limit extrapolation to bins below lowest good bin:
        t_at_l_bad = (t_at_l == fill) & (l_ind < lowest_good_ind)
        t_at_l[t_at_l_bad] = lowest_good[t_at_l_bad]

    return t

def _merra_land_mask(data):
    """ Convert fractional merra land mask to 1=land 0=ocean.

    XXX TODO: need to add in FRLANDICE so antarctica and greenland get included.
    """
    # UGH my design has officially fallen apart.
    mask_sd = Dataset(mask_file)
    frlandice = mask_sd.variables['FRLANDICE'][0] #0th time index
    data = frlandice + data
    return data > 0.25

def _hack_snow(data):
    """ Force greenland/antarctica to be snowy like CFSR """
    mask_sd = Dataset(mask_file)
    frlandice = mask_sd.variables['FRLANDICE'][0] # 0th time index
    data[frlandice > 0.25] = 100.0
    return data

def _trim_toa(data):
    if len(data.shape) != 3:
        print('Warning: why did you run _trim_toa on a non-3d var?')
    # at this point (before _reshape), data should be (level, lat, lon) and
    # the level dim should be ordered surface -> toa
    return data[0:len(LEVELS),:,:]

class MerraConversion(object):
    """ TODO doc """
    def __init__(self, in_dataset, in_name, out_name, out_units,
            units_fn, ndims_out):
        self.in_dataset = in_dataset
        self.in_name = in_name
        self.out_name = out_name
        self.out_units = out_units
        # this function will be applied to data for units conversion
        self.units_fn = units_fn
        self.ndims_out = ndims_out

    def do_conversion(self, sd, time_ind):
        # sd[self.in_dataset] = netCDF4 file handle
        # self.in_name = variable name
        in_sds = sd[self.in_dataset]
        print('BTH: performing data pull on',self.in_dataset,self.in_name)
        data = np.asarray(in_sds.variables[self.in_name])
        # BTH: data has changed from a netCDF4 variable object to a numpy array
        #      after this point, so checks on variable attributes (e.g. _FillValue) 
        #      need to be applied to in_sds.variables[self.in_name] 
        #      rather than to data. "MaskedArray object has no attribute <ex>" is a
        #      sign that the attribute read is being attempted on data rather than
        #      the native netCDF4 variable object.
        if self.in_name == 'lev':
            # insurance policy while levels are hard-coded in unit conversion fn's
            assert len(data) == 42
        ndims_in = len(data.shape)
        if ndims_in == 3 or ndims_in == 4:
            # note, vars w/ 3 spatial dims will be 4d due to time
            data = data[time_ind]
        if len(data.shape) == 3: # 3-dimensional; need to trim highest levels
            data = _trim_toa(data)
        # BTH: netCDF4 uses strings to track datatypes - need to do a conversion
        #      between the string and the equivalent SD.<DTYPE>
        nc4_dtype = data.dtype
        SD_dtype = nc4_to_SD_dtype(nc4_dtype)
        data_type = SD_dtype # int, float etc; mirror input type for now
        if self.out_name == 'pressure levels':
            data = data[0:len(LEVELS)].astype(np.float32) # trim to top CFSR level
            data_type = SDC.FLOAT32 # don't want double
        if self.out_name == 'total ozone':
            # b/c we vertically integrate ozone to get dobson units here
            shape = (data.shape[1], data.shape[2])
        elif self.ndims_out == 3:
            # clavr-x needs level to be the last dim
            shape = (data.shape[1], data.shape[2], data.shape[0])
        else:
            shape = data.shape
        out_sds = sd['out'].create(self.out_name, data_type, shape)
        out_sds.setcompress(SDC.COMP_DEFLATE, value=comp_level)
        self._set_dim_names(out_sds)
        if self.out_name == 'total ozone':
            # O3 field has _FillValue, missing_value, fmissing_value
            # but they are all the same value!
            fill = in_sds.variables[self.in_name]._FillValue
            data[data == fill] = np.nan
            dobson = kgkg_to_dobson(data)
            dobson[np.isnan(dobson)] = fill
            out_sds.set(_refill(_reshape(dobson, self.ndims_out, fill), fill))
        elif self.out_name == 'rh':
            temp_sds = in_sds.variables['T'] # temperature in [K] (Time, Height, Y, X)
            temp_k = temp_sds[time_ind]
            temp_k = _trim_toa(temp_k)
            fill = temp_sds._FillValue
            temp_k[temp_k == fill] = np.nan
            fill = in_sds.variables[self.in_name]._FillValue
            data[data == fill] = np.nan
            rh = qv_to_rh(data, temp_k)
            rh[np.isnan(rh)] = fill
            out_sds.set(_refill(_reshape(rh, self.ndims_out, fill), fill))
        elif self.out_name == 'rh at sigma=0.995':
            temp_sds = in_sds.variables['T10M'] # temperature in [K] (Time, Y, X)
            temp_k = temp_sds[time_ind]

            ps_sds = in_sds.variables['PS'] # surface pressure in [Pa]
            fill = ps_sds._FillValue
            ps_pa = ps_sds[time_ind]
            ps_pa[ps_pa == fill] = np.nan

            rh = qv_to_rh(data, temp_k, ps=ps_pa)
            rh[np.isnan(rh)] = fill
            out_sds.set(_refill(_reshape(rh, self.ndims_out, fill), fill))
        else:
            if '_FillValue' in in_sds.variables[self.in_name].ncattrs():
                fill = in_sds.variables[self.in_name]._FillValue
                if self.out_name == 'water equivalent snow depth':
                    # Special case: set snow depth missing values to 0
                    # to match CFSR behavior.
                    data[data == fill] = 0.0
                    converted = _hack_snow(data)
                else:
                    converted = self.units_fn(data)
                    converted[data == fill] = fill
                out_sds.set(_refill(_reshape(converted, self.ndims_out, fill), fill))
            elif 'missing_value' in in_sds.variables[self.in_name].ncattrs():
                fill = in_sds.variables[self.in_name].missing_value
                if self.out_name == 'water equivalent snow depth':
                    # Special case: set snow depth missing values to 0 
                    # to match CFSR behavior.
                    data[data == fill] = 0.0
                    converted = _hack_snow(data)
                else:
                    fill = in_sds.variables[self.in_name].missing_value
                    converted = self.units_fn(data)
                    converted[data == fill] = fill
                out_sds.set(_refill(_reshape(converted, self.ndims_out, fill), fill))
            else:
                # no need to _refill
                if self.out_name == 'pressure levels':
                    data = data[::-1] # clavr-x needs toa->surface
                if self.out_name == 'lon':
                    tmp = np.copy(data)
                    n = data.shape[0]
                    halfway = n // 2
                    data[0:halfway] = tmp[halfway:]
                    data[halfway:] = tmp[0:halfway]
                out_sds.set(_reshape(self.units_fn(data), self.ndims_out, None))
        try:
            out_sds.setfillvalue(CLAVRX_FILL)
            if self.out_units is not None:
                setattr(out_sds, 'units', self.out_units)
            if 'units' in sd[self.in_dataset].variables[self.in_name].ncattrs():
                u = ' in [' + sd[self.in_dataset].variables[self.in_name].units + ']'
            else:
                u = ''
            setattr(out_sds, 'source_data', 
                    'MERRA->' + self.in_dataset + '->' + self.in_name + u)
            setattr(out_sds, 'long_name', sd[self.in_dataset].variables[self.in_name].long_name)
            #setattr(out_sds, 'missing_value', in_sds.attributes()['missing_value'])
            # not sure about diff. btwn missing_value and fmissing_value
            #setattr(out_sds, 'fmissing_value', in_sds.attributes()['fmissing_value'])
        except KeyError:
            # dims will fail...  XXX don't want failed attribute to stop other attributes!
            pass
        out_sds.endaccess()

    def _set_dim_names(self, out_sds):
        if self.in_name == 'lat':
            out_sds.dim(0).setname('lat')
        elif self.in_name == 'lon':
            out_sds.dim(0).setname('lon')
        elif self.in_name == 'lev':
            out_sds.dim(0).setname('level')
        elif self.ndims_out == 2:
            out_sds.dim(0).setname('lat')
            out_sds.dim(1).setname('lon')
        elif self.ndims_out == 3:
            out_sds.dim(0).setname('lat')
            out_sds.dim(1).setname('lon')
            out_sds.dim(2).setname('level')
        else:
            raise ValueError("unsupported dimensionality")

def make_merra_one_day(in_files, out_dir, mask_file):
    """ TODO doc """
    sd = dict()
    for k in in_files.keys():
        sd[k] = Dataset(in_files[k])

    try:
        # --- build a list of all times in all input files
        times = dict()
        time_set = dict()
        for k in in_files.keys():
            times[k] = []
            time_set[k] = set()
            if 'time' in sd[k].variables.keys():
                TIME_HACK_OFFSET = 0
                t_sds = sd[k].variables['time']
                t_units = t_sds.units # expect format "minutes since %Y-%m-%d %H:%M:%S"
                base_time = datetime.strptime(t_units + ' UTC', 'minutes since %Y-%m-%d %H:%M:%S %Z')
                # A non-zero TIME_HACK_OFFSET is going to be equal to base_time.minute
                TIME_HACK_OFFSET = base_time.minute
            else:
                raise ValueError("Couldn't find time coordinate in this file")
            for (i,t) in enumerate(t_sds):
                if t_units.startswith('minutes'):
                    time = base_time + timedelta(minutes=int(t))
                elif t_units.startswith('hours'):
                    time = base_time + timedelta(hours=int(t))
                else:
                    raise ValueError("Can't handle time unit")
                if time.minute == TIME_HACK_OFFSET:
                    # total hack to deal with non-analysis products being on the half-hour 
                    time = time - timedelta(minutes=TIME_HACK_OFFSET)
                # This is just to get the index for a given timestamp later:
                times[k].append((i, time))
                time_set[k].add(time)
        # find set of time common to all input files
        common_times = (time_set['ana'] & 
                        time_set['flx'] & 
                        time_set['slv'] &
                        time_set['lnd'] &
                        time_set['asm3d'] &
                        time_set['asm2d'] &
                        time_set['rad'])

        # if we don't have 4 common times something is probably terribly wrong
        assert len(common_times) == 4

        out_fnames = []
        for out_time in common_times:
            print('    working on time: {}'.format(out_time))
            out_fname = os.path.join(out_dir, out_time.strftime('merra.%y%m%d%H_F000.hdf'))
            out_fnames.append(out_fname)
            sd['out'] = SD(out_fname, SDC.WRITE|SDC.CREATE|SDC.TRUNC) # TRUNC will clobber existing

            # --- determine what time index we want from input files
            time_inds = dict()
            for k in in_files.keys():
                # Get the index for the current timestamp:
                time_inds[k] = [i for (i, t) in times[k] if t == out_time][0]

            # --- write out data variables
            for k in rs:
                rsk = rs[k]
                MerraConversion(
                            rsk['in_file'],
                            rsk['in_varname'],
                            k,
                            rsk['out_units'],
                            rsk['units_fn'], 
                            rsk['ndims_out']
                        ).do_conversion(sd, time_inds[rsk['in_file']])

            # --- handle surface height and static land mask from constants (mask_file) specially
            sd['mask'] = Dataset(mask_file)
            MerraConversion(
                        'mask', 
                        'PHIS',
                        'surface height',
                        'km',
                        lambda a: a / 9806.65, # Convert geopotential (m^2 s^-2) to geopotential height via h/(1000.*g)
                        2
                    ).do_conversion(sd, 0)
            MerraConversion(
                        'mask', 
                        'FRLAND',
                        'land mask',
                        '1=land, 0=ocean, greenland and antarctica are land',
                        _merra_land_mask,
                        2
                    ).do_conversion(sd, 0)
            # --- handle ice-fraction and land ice-fraction from constants (mask_file) specially
            MerraConversion(
                        'mask',
                        'FRACI',
                        'FRACI',       # We are using FRSEAICE for ice-fraction, since GFS uses a sea-ice fraction as 'ice fraction'. This version of ice-fraction is broken, so it is being shielded from CLAVR-x use with the output name 'FRACI' until this gets figured out.
                        'none',
                        no_conversion,
                        2
                    ).do_conversion(sd, 0)
            MerraConversion(
                        'mask',
                        'FRLANDICE',
                        'land ice fraction',
                        'none',
                        no_conversion,
                        2
                    ).do_conversion(sd, 0)


            # --- write global attributes
            var = sd['out'].select('temperature')
            nlevel = var.dimensions(full=False)['level']
            nlat = var.dimensions(full=False)['lat']
            nlon = var.dimensions(full=False)['lon']
            setattr(sd['out'], 'NUMBER OF LATITUDES', nlat)
            setattr(sd['out'], 'NUMBER OF LONGITUDES', nlon)
            setattr(sd['out'], 'NUMBER OF PRESSURE LEVELS', nlevel)
            setattr(sd['out'], 'NUMBER OF O3MR LEVELS', nlevel)
            setattr(sd['out'], 'NUMBER OF RH LEVELS', nlevel)
            setattr(sd['out'], 'NUMBER OF CLWMR LEVELS', nlevel)
            lat = sd['out'].select('lat')
            lon = sd['out'].select('lon')
            attr = sd['out'].attr('LATITUDE RESOLUTION')
            attr.set(SDC.FLOAT32, lat.get()[1] - lat.get()[0])
            attr = sd['out'].attr('LONGITUDE RESOLUTION')
            attr.set(SDC.FLOAT32, lon.get()[1] - lon.get()[0])
            attr = sd['out'].attr('FIRST LATITUDE')
            attr.set(SDC.FLOAT32, lat.get()[0])
            attr = sd['out'].attr('FIRST LONGITUDE')
            attr.set(SDC.FLOAT32, lon.get()[0])
            setattr(sd['out'], 'GRIB TYPE', 'not applicable') # XXX better to just not write this attr?
            setattr(sd['out'], '3D ARRAY ORDER', 'ZXY') # XXX is this true here?
            [a.endaccess() for a in [var, lat, lon]]

            sd['out'].end()

    finally:
        for k in in_files.keys():
            #sd[k].end()
            print('Finished',k)

    return out_fnames

if __name__ == '__main__':
    #inpath = '/fjord/jgs/personal/mfoster/MERRA/'
    #outpath = '/fjord/jgs/personal/mhiley/MERRA/'
    #inpath = '/Volumes/stuff/merra/input/'
    #outpath = '/Volumes/stuff/merra/output/'
    #inpath = '/home/clavrx_ops/clavrx_MERRA2/merra2/tmp/'
    #outpath = '/home/clavrx_ops/clavrx_MERRA2/merra2/tmp/out/'
    inpath = '/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/MERRA_INPUT/tmp/'
    #outpath = '/data/Personal/joleenf/test_BH_merra2/clavrx_ancil_data/dynamic/merra2'
    outpath = "/apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2"

    try:
        date_str_arg = sys.argv[1]
        date_parsed = datetime.strptime(date_str_arg, '%Y%m%d')
    except:
        print('usage:\n    python merra4clavrx.py 20090101')
        exit()

    year_str = date_str_arg[0:4]
    outpath_full = os.path.join(outpath, year_str) + '/'
    #inpath_full = inpath
    inpath_full = os.path.join(inpath, year_str)

    try:
        os.makedirs(outpath_full)
    except OSError:
        pass # dir already exists
    # BTH: Define mask_file here
    print("looking at {}".format(inpath_full + '/2d_ctm/MERRA2_101.const_2d_ctm_Nx.'+date_str_arg+'.nc4'))
    mask_file = glob(inpath_full + '/2d_ctm/MERRA2_101.const_2d_ctm_Nx.'+date_str_arg+'.nc4')[0]
    print('Processing date: {}'.format(date_parsed.strftime('%Y-%m-%d')))
    in_files = { 
            'ana': glob(inpath_full + '/3d_ana/MERRA2*ana_Np.' + 
                date_str_arg + '.nc4')[0],
            'flx': glob(inpath_full + '/2d_flx/MERRA2*flx_Nx.' + 
                date_str_arg + '.nc4')[0],
            'slv': glob(inpath_full + '/2d_slv/MERRA2*slv_Nx.' + 
                date_str_arg + '.nc4')[0],
            'lnd': glob(inpath_full + '/2d_lnd/MERRA2*lnd_Nx.' + 
                date_str_arg + '.nc4')[0],
            'asm3d': glob(inpath_full + '/3d_asm/MERRA2*asm_Np.' +
                date_str_arg + '.nc4')[0],
            'asm2d': glob(inpath_full + '/2d_asm/MERRA2*asm_Nx.' +
                date_str_arg + '.nc4')[0],
            'rad': glob(inpath_full + '/2d_rad/MERRA2*rad_Nx.' +
                date_str_arg + '.nc4')[0],
        }
    out_files = make_merra_one_day(in_files, outpath_full, mask_file)
    print('out_files: {}'.format(list(map(os.path.basename, out_files))))
