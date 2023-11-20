def quantile_stats(in_array):
    """Return Quantile for Data Array."""
    return np.nanquantile(in_array, Q_ARR)


def write_debug(data_array, where_msg, var_name):
    """For debugging purposes, print quartiles at various points in program."""
    if isinstance(data_array, np.ma.core.MaskedArray):
        data_array = np.ma.filled(data_array, np.nan)
    quantiles = quantile_stats(data_array)
    quantiles_string = "[{}]".format(", ".join(map(str, quantiles)))
    txt = "{} {} \n {}".format(var_name, where_msg, quantiles_string)
    LOG.info(txt)
    return quantiles
