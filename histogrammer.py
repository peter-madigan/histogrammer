'''
General approach:
The user defines:
 1. a dataset path (and optionally a variable in that dataset)
 2. optional, the bin edges
 3. a collection of variables and filters defined by:
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

If a batch_size is defined, then the files will be looped over rather than
loading all data at once.

Yaml spec::

    histograms:
        <histogram name>:
          variable: <variable name>
          field: <str, optional>
          bins: <[bin_low_edge, ..., bin_high_edge] or
            low: bin_low_edge, high: bin_high_edge, n: nbins>

    variables:
      <variable name>:
        path:
          - <dataset 0 name>
          - ...
          - <dataset N name>
        field: <str, optional>

    filters:
      <filter name>: <str describing filter>

    import:
     - <module name>
'''
import os
import multiprocessing
import argparse
import warnings

import h5py
import yaml
import tqdm
import numpy as np

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


def generate_histogram(i, filepath, variable, bins, field=None, variables=None, filters=None, batch_size=None, imports=None, loop=None):
    if imports is not None:
        for lib in imports:
            globals()[lib] = __import__(lib)

    # Open the file
    f = H5FlowDataManager(filepath, 'r', mpi=False)

    # Initialize histogram(s)
    bins = [generate_bins(b) for b in bins]
    hist = dict()
    hist['overall'] = np.zeros([len(b)-1 for b in bins])
    if filters:
        for filt in filters:
            hist[filt] = np.zeros([len(b)-1 for b in bins])

    # Initialize loop
    if loop is None:
        rows = f[variables[variable[0]]['path'][0] + '/data'].shape[0]
    else:
        rows = f[variables[loop]['path'][0] + '/data'].shape[0]
    if batch_size:
        batches = tqdm.tqdm([slice(i,i+batch_size) for i in range(0,rows,batch_size)],
            position=i, smoothing=0, desc=os.path.basename(filepath))
    else:
        batches = tqdm.tqdm([slice(None)],
            position=i, desc=os.path.basename(filepath))

    # Initialize filter operations
    filt_op = dict()
    if filters:
        for filt,expr in filters.items():
            filt_op[filt] = eval('lambda : (' + expr +').astype(bool)')

    # Run loop
    for batch in batches:
        # Load data
        if variables is not None:
            for var,spec in variables.items():
                # Load variable
                try:
                    globals()[var] = f[tuple(spec['path'] + [batch])]
                    var_field = spec.get('field')
                    if var_field:
                        globals()[var] = globals()[var][var_field]
                except Exception as e:
                    warnings.warn(f'error in {var} : '+str(e))
                    globals()[var] = None

        # Apply filters
        if filters:
            for filt,spec in filters.items():
                # Apply filter function
                try:
                    globals()[filt] = filt_op[filt]()
                except Exception as e:
                    warnings.warn(f'error in {filt} : '+str(e))
                    globals()[filt] = slice(None)

        data = [globals()[v] for v in variable]
        if field:
            data = [np.clip(d[f].ravel(), b[0], b[-1]) if f is not None else np.clip(d.ravel(), b[0], b[-1]) for d,f,b in zip(data, field, bins)]

        # Update histograms
        hist['overall'] += np.histogramdd(data, bins=bins)[0]
        if filters:
            for filt in filters:
                hist[filt] += np.histogramdd([d[globals()[filt].reshape(d.shape)] for d in data], bins=bins)[0]

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
        for filt in hist_dict:
            if filt not in run_grp:
                run_grp.create_dataset(filt, data=hist_dict[filt])
            else:
                run_grp[filt][:] = run_grp[filt][:] + hist_dict[filt]

        # update global histograms
        sum_grp = f[hist_name]['sum']
        for filt in hist_dict:
            if filt not in sum_grp:
                sum_grp.create_dataset(filt, data=hist_dict[filt])
            else:
                sum_grp[filt][:] = sum_grp[filt][:] + hist_dict[filt]


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
            print(f'Generating {hist}...')
            results = []
            for i,filepath in enumerate(filepaths):
                results.append(p.apply_async(generate_histogram,
                    tuple(),
                    dict(
                        i=i % processes, filepath=filepath,
                        variable=config['histograms'][hist]['variable'],
                        bins=config['histograms'][hist]['bins'],
                        field=config['histograms'][hist].get('field'),
                        variables=config.get('variables'),
                        filters=config.get('filters'),
                        batch_size=batch_size,
                        loop=config['histograms'][hist].get('loop'),
                        imports=config.get('import'))))

            for filepath, result in zip(filepaths, results):
                save(outpath, hist, filepath, result.get(), config['histograms'][hist])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
