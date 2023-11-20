def printValue(data, x=180, y=360, z=slice(None,None,1)):
    print(data.dims)
    if "longitude" in data.dims:
        print(data.isel(longitude=x, latitude=y, isobaricInhPa=z).data)
    elif "lon" in data.dims:
        print(data.isel(lon=x, lat=y, isobaricInhPa=z).data)
    else:
        print(data.isel(x=x, y=y, z=z).data)
