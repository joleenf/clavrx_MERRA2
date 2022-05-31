#!/bin/bash


nyear=${1:-1998}
field=${2:-rh}

if [ "${nyear}" == "-h" ]; then
	`/bin/pod2usage $0`
	exit
fi

filepath=$HOME/data/archive_quick_stats
repo_dir=$HOME/clavrx_MERRA2

mkdir -p $filepath

output_file=$filepath/${field}_${nyear}.txt

for fn in `ls /apollo/cloud/Ancil_Data/clavrx_ancil_data/dynamic/merra2/$nyear/*`;do python $repo_dir/python/list_stats.py $fn $field;done >> $output_file 

	: <<=cut
=pod

=head1 NAME


    quick_stats.sh

=head1 SYNOPSIS

    sh quick_stats.sh year field

    Parameters:
        year:  YYYY year of data in which listing of min/max field will be generated
        field:  Field for which min/max will be generated from the MERRA2 product files

=head1 DESCRIPTION
    Quick stats to check min/max of mainly the rh field for MERRA2 in each year that it is run.
    Text file will be placed in $HOME/data/archive_quick_stats/<field>_<year>.txt where year
    and field match the input variables.

=head2 Requirements

=cut
