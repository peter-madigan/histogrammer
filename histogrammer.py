'''
This is a tool to assist in combining data from a large number of runs
to generate histograms of various parameters of interest.

For each histogram, a user defines:
 1. a particular variable used to fill the histogram (it can be multiple for N-dimensional histograms)
 2. the bins used for the histogram
 3. a set of filters and variables defining the unique histograms to be generated
 3. a path to datasets used create the histograms, variables, and filters
  a. a dataset path (and optionally a variable in that dataset)
  b. a python expression that returns an array of bools

This script will then generate a file containing for each input filename:
    - the overall histogram
    - the histogram broken down by each filter
along with the sum across all files.

Group structure in output HDF5 file::

    <histogram name>/sum/overall
    <histogram name>/sum/<filter name>
    <histogram name>/run/<filename>/overall
    <histogram name>/run/<filename/<filter name>

Bins will be saved as attributes (`bins{i}`) under the `<histogram>` group along
with a string of the configuration used to generate the histogram.

If a ``--batch_size`` is defined, then the files will be looped over rather than
loading all data at once.

Yaml spec::

    import: # optional
     - <python module name required by variables>

    histograms:
        <histogram name>:
          variable: <variable name> or [<var 1 name>, <var 2 name>, ...]
          field: <str or null, optional> or [<field 1 name>, <field 2 name>, ...]
          loop: <variable used to calculate loop size, optional>
          bins: [<bin_low_edge>, ..., <bin_high_edge>] or
            {low: bin_low_edge, high: bin_high_edge, n: nbins} or list of either for N-D histograms

    variables:
      <variable name>:
        expr: <eval string describing variable>
        filt: <bool, true if a histogram should be generated with this variable interpreted as a filter, optional, default=True>

    datasets:
      <dset name>:
        path:
          - <dataset 0 name>
          - ...
          - <dataset N name>
        field: <str, optional>

'''
import os
import multiprocessing
import argparse
import warnings

import h5py
import yaml
import tqdm
import numpy as np
import numpy.ma as ma

from h5flow import H5FLOW_MPI
global H5FLOW_MPI
H5FLOW_MPI = False
from h5flow.data import H5FlowDataManager


def generate_bins(spec):
    if isinstance(spec, dict):
        bins = np.linspace(spec['low'], spec['high'], spec['n']+1)
    else:
        bins = np.array(spec)
    return bins


def only_valid(maybe_masked_arr):
    if maybe_masked_arr is None:
        return None
    return (maybe_masked_arr.ravel()
        if not ma.is_masked(maybe_masked_arr)
        else maybe_masked_arr.compressed())


def generate_histogram(i, filepath, variable, bins, field=None, weight=None, datasets=None, variables=None, batch_size=None, imports=None, loop=None):
    if imports is not None:
        for lib in imports:
            globals()[lib] = __import__(lib)

    # Open the file
    f = H5FlowDataManager(filepath, 'r', mpi=False)

    # Initialize histogram(s)
    bins = [generate_bins(b) for b in bins]
    hist = dict()
    hist['overall'] = np.zeros([len(b)-1 for b in bins])
    if variables:
        for var in variables:
            if variables[var].get('filt',True):
                hist[var] = np.zeros([len(b)-1 for b in bins])

    # Initialize loop
    if loop is None:
        rows = f[datasets[variable[0]]['path'][0] + '/data'].shape[0]
    else:
        rows = f[datasets[loop]['path'][0] + '/data'].shape[0]
    if batch_size:
        batches = tqdm.tqdm([slice(i,i+batch_size) for i in range(0,rows,batch_size)],
            position=i, smoothing=0, desc=os.path.basename(filepath))
    else:
        batches = tqdm.tqdm([slice(0,rows)],
            position=i, desc=os.path.basename(filepath))

    # Initialize filter operations
    var_op = dict()
    if variables:
        for var in variables:
            expr = variables[var]['expr']
            var_op[var] = eval('lambda : ' + expr)

    # Run loop
    for batch in batches:
        # Load data
        if datasets is not None:
            for dset,spec in datasets.items():
                # Load dataset
                try:
                    globals()[dset] = f[tuple(spec['path'] + [batch])]
                    dset_field = spec.get('field')
                    if dset_field:
                        globals()[dset] = globals()[dset][dset_field]
                except Exception as e:
                    warnings.warn(f'error in {dset} : '+str(e))
                    globals()[dset] = None

        # Load variables
        if variables:
            for var in variables:
                # Apply variable function
                try:
                    globals()[var] = var_op[var]()
                except Exception as e:
                    warnings.warn(f'error in {var} : '+str(e))
                    globals()[var] = slice(None)

        data = [globals()[v] for v in variable]
        if field:
            data = [d[f].clip(b[0], b[-1]) if f is not None else d.clip(b[0], b[-1]) for d,f,b in zip(data, field, bins)]

        # Update histograms
        if weight:
            w = globals()[weight]
        else:
            w = None
        hist['overall'] += np.histogramdd([only_valid(d) for d in data], bins=bins, weights=only_valid(w))[0]
        if variables:
            for var in variables:
                if variables[var].get('filt', True) and np.any(globals()[var]):
                    mask = globals()[var].astype(bool)
                    hist[var] += np.histogramdd([only_valid(d[mask]) for d in data], bins=bins, weights=only_valid(w[mask] if w is not None else None))[0]

    # Return histograms
    return hist.copy()


def save(outpath, hist_name, filepath, hist_dict, hist_config):
    with h5py.File(outpath, 'a') as f:
        # maybe create new histogram entry
        if hist_name not in f:
            f.create_group(hist_name)
            f[hist_name].attrs['config'] = str(hist_config)
            for i,b in enumerate(hist_config['bins']):
                f[hist_name].attrs[f'bins{i}'] = generate_bins(b)

            f[hist_name].create_group('sum')
            f[hist_name].create_group('run')

        # maybe create new individual run dataset
        filename = os.path.basename(filepath)
        if filename not in f[hist_name]['run']:
            f[hist_name]['run'].create_group(filename)
            f[hist_name]['run'][filename].attrs['filepath'] = filepath

        # update run-level histograms
        run_grp = f[hist_name]['run'][filename]
        for hist in hist_dict:
            if hist not in run_grp:
                run_grp.create_dataset(hist, data=hist_dict[hist])
            else:
                run_grp[hist][:] = run_grp[hist][:] + hist_dict[hist]

        # update global histograms
        sum_grp = f[hist_name]['sum']
        for hist in hist_dict:
            if hist not in sum_grp:
                sum_grp.create_dataset(hist, data=hist_dict[hist])
            else:
                sum_grp[hist][:] = sum_grp[hist][:] + hist_dict[hist]


def main(config_yaml, outpath, filepaths, processes=None, batch_size=None, **kwargs):
    # load configuration
    with open(config_yaml) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    # parse config
    for hist_name in list(config['histograms'].keys()):
        hist_config = config['histograms'][hist_name]
        if not isinstance(hist_config['variable'], list):
            hist_config['variable'] = [hist_config['variable']]
            if 'field' in hist_config:
                hist_config['field'] = [hist_config['field']]
            if 'bins' in hist_config:
                hist_config['bins'] = [hist_config['bins']]

    with h5py.File(outpath, 'a') as f:
        f['/'].attrs['config'] = str(config)

    processes = processes if processes is not None else multiprocessing.cpu_count()
    processes = min(processes, len(filepaths))

    print(f'Running on {processes} processes...')
    with multiprocessing.Pool(processes) as p:
        for hist in config.get('histograms', dict()):
            print(f'\nGenerating {hist}...')
            results = []
            for i,filepath in enumerate(filepaths):
                results.append(p.apply_async(generate_histogram,
                    tuple(),
                    dict(
                        i=i % processes, filepath=filepath,
                        variable=config['histograms'][hist]['variable'],
                        bins=config['histograms'][hist]['bins'],
                        field=config['histograms'][hist].get('field'),
                        datasets=config.get('datasets'),
                        variables=config.get('variables'),
                        batch_size=batch_size,
                        loop=config['histograms'][hist].get('loop'),
                        weight=config['histograms'][hist].get('weight'),
                        imports=config.get('import'))))

            for filepath, result in zip(filepaths, results):
                save(outpath, hist, filepath, result.get(), config['histograms'][hist])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outpath', '-o', type=str,
        required=True, help='''output hdf5 file to generate''')
    parser.add_argument('--filepaths', '-i', nargs='+', type=str, required=True,
        help='''input h5flow files to generate histograms from''')
    parser.add_argument('--config_yaml', '-c', type=str, required=True,
        help='''yaml config file''')
    parser.add_argument('--processes', '-p', type=int, default=None,
        help='''number of parallel processes (defaults to number of cpus detected)''')
    parser.add_argument('--batch_size', '-b', type=int, default=None,
        help='''batch size for loop (optional)''')
    args = parser.parse_args()

    main(**vars(args))
