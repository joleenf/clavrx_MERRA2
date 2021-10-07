You may see a warning of the form:

"UserWarning: WARNING: valid_range not used since it cannot be safely cast to variable data type"

This is not a fatal warning. It just means that when python pulls data from the netCDF file during
the enumerate() function, it does not understand the .valid_range attribute (which is used to
mask data) and therefore it isn't carrying it along. Since the time information in the enumerate()
should not be masked to begin with, this *shouldn't* be a concern.
