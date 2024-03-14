## [GEOS-5 List of Files Downloaded and products used.](https://fluid.nccs.nasa.gov/weather/)
NASA Office Note Repository: https://gmao.gsfc.nasa.gov/pubs/office_notes.php
GMAO Office Note No. 4 (Version 1.2): Lucchesi, R., 2018. File Specification for GEOS-5 FP (Forward Processing)
https://gmao.gsfc.nasa.gov/pubs/docs/Lucchesi1203.pdf

### const_2d_asm_Nx
Constants File

https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/GEOS.fp.asm.const_2d_asm_Nx.00000000_0000.V01.nc4

### inst3_2d_asm_Nx
2d assimilated state

| Product Varname | Product Description |
| ----------- | ----------- |
| TQV | total precipitable water vapor |

### inst3_3d_asm_Np
3d assimilated state on pressure levels
<br>(available with Forecast Data)

| Product Varname | Product Description |
| ----------- | ----------- |
| SLP | MSL Pressure |
| T | Temperature |
| PS | Surface Pressure |
| H | Height |
| U | U-wind |
| V | V-wind |
| QI | Mass Fraction of Cloud Ice Water |
| QL | mass fraction of cloud liquid water |
| QV | Relative Humidity |
| O3 | Ozone Mass Mixing Ratio (Total Ozone in final hdf) |

### tavg1_2d_flx_Nx
2d time-averaged surface flux diagnostics
<br>(available with Forecast Data)

| Product Varname | Product Description |
| ----------- | ----------- |
| PBLH | Planetary Boundary Layer Height |
| FRSEAICE | Ice Covered Fraction of Tile |

### tavg1_2d_lnd_Nx
2d time-averaged land surface diagnostics
<br>(available with Forecast Data)

| Product Varname | Product Description |
| ----------- | ----------- |
| SNOWMAS | Total snow storage land in kg/m2 |

### tavg1_2d_slv_Nx
2d time-averaged single level diagnostics
<br>(available with Forecast Data)

| Product Varname | Product Description |
| ----------- | ----------- |
| TROPPT | tropopause pressure based on thermal estimate |
| TROPT | tropopause temperature using blended TROPP estimate |
| U50M | eastward wind at 50 meters |
| V50M | northward wind at 50 meters m |
| TS | surface skin temperature |
 | TQV | Total Precipitable Water Vapor |
| QV10M | 10-meter specific humidity used as rh at sigma=0.995 |
| T10M | 10-meter air temperature used as temperature at sigma=0.995 |
| V10M | 10-meter eastward wind used as v-wind at sigma=0.995 |
| U10M | 10-meter northward wind used as u-wind at sigma=0.995 |


### tavg1_2d_rad_Nx
2d time-averaged radiation diagnostics
<br>(available with Forecast Data)

| Product Varname | Product Description |
| ----------- | ----------- |
| CLDTOT | total cloud area fraction |
