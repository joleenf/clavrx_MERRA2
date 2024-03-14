data = {
    # --- data vars from 'inst3_3d_asm_Np'
    'MSL pressure': {
        'in_file': 'asm3d',
        'in_varname': 'SLP',
        'out_units': 'hPa',
        'units_fn': lambda a: a/100.0, # scale factor for Pa --> hPa
        'ndims_out': 2
        },
    'temperature': {
        'in_file': 'asm3d',
        'in_varname': 'T',
        'out_units': 'K',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 3
        },
    'surface pressure': {
        'in_file': 'asm3d',
        'in_varname': 'PS',
        'out_units': 'hPa',
        'units_fn': lambda a: a/100.0, # scale factor for Pa --> hPa
        'ndims_out': 2
        },
    'height': { 
        'in_file': 'asm3d',
        'in_varname': 'H',
        'out_units': 'km',
        'units_fn': lambda a: a/1000.0, # scale factor for m --> km
        'ndims_out': 3
        },
    'u-wind': {
        'in_file': 'asm3d',
        'in_varname': 'U',
        'out_units': 'm/s',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 3
        },
    'v-wind': {
        'in_file': 'asm3d',
        'in_varname': 'V',
        'out_units': 'm/s',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 3
        },
    'rh': {
        'in_file': 'asm3d',
        'in_varname': 'QV',
        'out_units': '%',
        'units_fn': None, # special case due to add'l inputs
        'ndims_out': 3
        },
    'total ozone': {
        'in_file': 'asm3d',
        'in_varname': 'O3',
        'out_units': 'Dobson',
        'units_fn': None, # special case due to add'l inputs
        'ndims_out': 2
        },
    'o3mr': {
        'in_file': 'asm3d',
        'in_varname': 'O3',
        'out_units': 'kg/kg',
        'units_fn': "!!python/name:conversions.no_conversion",
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
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    'u-wind at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'U10M',
        'out_units': 'm/s',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    'v-wind at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'V10M',
        'out_units': 'm/s',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    'surface temperature': { # XXX confirm skin temp is correct choice for 'surface temperature'
        'in_file': 'slv',
        'in_varname': 'TS',
        'out_units': 'K',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    'temperature at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'T10M',
        'out_units': 'K',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    'rh at sigma=0.995': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'QV10M',
        'out_units': '%',
        'units_fn': "!!python/name:conversions.no_conversion", # XXX how to get p at sigma=0.995 for RH conversion?
        'nan_fill': True,
        'ndims_out': 2
        },
    'u-wind at 50M': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'U50M',
        'out_units': 'm/s',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    'v-wind at 50M': { # not actually exactly sigma=0.995???
        'in_file': 'slv',
        'in_varname': 'V50M',
        'out_units': 'm/s',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    # --- data vars from 'tavg1_2d_(flx)_Nx'
    'planetary boundary layer height': {
        'in_file': 'flx',
        'in_varname': 'PBLH',
        'out_units': 'km',
        'units_fn': "lambda a: a/1000.0, # scale factor for m --> km",
        'ndims_out': 2
        },
    'ice fraction': {
        'in_file': 'flx',
        'in_varname': 'FRSEAICE',
        'out_units': 'none',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    # --- data vars from 'tavg1_2d_(lnd)_Nx'
    'water equivalent snow depth': {
        'in_file': 'lnd',
        'in_varname': 'SNOMAS',
        'out_units': 'kg/m^2',
        'units_fn': "!!python/name:conversions.no_conversion", # special case in do_conversion will set fill values to zero
        'ndims_out': 2
        },
    # --- data vars from 'inst3_3d_(asm)_Np'
    'clwmr': {
        'in_file': 'asm3d',
        'in_varname': 'QL',
        'out_units': 'kg/kg',
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 3
        },
    'cloud ice water mixing ratio': {
        'in_file': 'asm3d',
        'in_varname': 'QI',
        'out_units': 'kg/kg',
        'units_fn': "!!python/name:conversions.no_conversion",
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
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 2
        },
    # --- geoloc vars from 'asm3d'
    'lon': {
        'in_file': 'asm3d',
        'in_varname': 'lon',
        'out_units': None,
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 1
        },
    'lat': {
        'in_file': 'asm3d',
        'in_varname': 'lat',
        'out_units': None,
        'units_fn': "!!python/name:conversions.no_conversion",
        'ndims_out': 1
        },
    'pressure levels': {
        'in_file': 'asm3d',
        'in_varname': 'lev',
        'out_units': 'hPa',
        'units_fn': "!!python/name:conversions.no_conversion", # yes this is indeed correct.
        'ndims_out': 1
        },
    }

import yaml
with open('geosfp.yml', 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
