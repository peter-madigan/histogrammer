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
import time

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
        if not 'log' in spec or spec['log'] == False:
            bins = np.linspace(spec['low'], spec['high'], spec['n']+1)
        else:
            bins = np.geomspace(spec['low'], spec['high'], spec['n']+1)            
    else:
        bins = np.array(spec)
    return bins


def only_valid(maybe_masked_arr, fill=0):
    if maybe_masked_arr is None:
        return None
    return (maybe_masked_arr.ravel()
            if not ma.is_masked(maybe_masked_arr)
            else maybe_masked_arr.filled(fill))


def generate_histograms(index, filepath, histograms, *args, datasets=None, variables=None, batch_size=None, imports=None, create_event_list=False, **kwargs):
    if imports is not None:
        for lib in imports:
            globals()[lib] = __import__(lib)

    # Open the file
    try:
        f = H5FlowDataManager(filepath, 'r', mpi=False)
        basename = os.path.basename(filepath)

        # Initialize histogram(s)
        bins = dict()
        hists = dict()
        for hist in histograms:
            bins[hist] = [generate_bins(b) for b in histograms[hist]['bins']]
            hists[hist] = dict()
            hists[hist]['overall'] = np.zeros([len(b)-1 for b in bins[hist]])
            if variables:
                for var in variables:
                    if variables[var].get('filt',True):
                        hists[hist][var] = np.zeros([len(b)-1 for b in bins[hist]])

        # Initialize loop
        loop_dataset = [datasets[d] for d in datasets if 'loop' in datasets[d] and datasets[d]['loop']][0]
        rows = f[loop_dataset['path'][0] + '/data'].shape[0]
        if batch_size:
            batches = [slice(i,min(i+batch_size,rows)) for i in range(0,rows,batch_size)]
        else:
            batches = [slice(0,rows)]

        # Initialize filter operations
        var_op = dict()
        if variables:
            for var in variables:
                expr = variables[var]['expr']
                var_op[var] = eval('lambda : ' + expr)

        # Run loop
        event_list = dict()
        if variables and create_event_list:
            for var in variables:
                if variables[var].get('filt', True):
                    event_list[var] = []
                
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
                        warnings.warn(f'error in {basename}/{dset} : '+str(e))
                        globals()[dset] = None

            # Load variables
            if variables:
                for var in variables:
                    # Apply variable function
                    try:
                        globals()[var] = var_op[var]()
                        if create_event_list and variables[var].get('filt', True):
                            event_list[var] += (np.where(globals()[var])[0] + batch.start).tolist()
                    except Exception as e:
                        warnings.warn(f'error in {basename}/{var} : '+str(e))
                        globals()[var] = slice(None)

            # Update histograms
            for hist in histograms:
                try:
                    variable = histograms[hist]['variable']
                    field = histograms[hist]['field']
                    weight = histograms[hist]['weight']
            
                    data = [globals()[v] for v in variable]
                    if field:
                        data = [d[f].clip(b[0], b[-1]) if f is not None else d.clip(b[0], b[-1]) for d,f,b in zip(data, field, bins[hist])]

                    if weight:
                        w = globals()[weight]
                    else:
                        w = None

                    hists[hist]['overall'] += np.histogramdd([only_valid(d, bins[hist][i][0]) for i,d in enumerate(data)], bins=bins[hist], weights=only_valid(w))[0]
                    if variables:
                        for var in variables:
                            if variables[var].get('filt', True) and np.any(globals()[var]):
                                mask = globals()[var].astype(bool)
                                hists[hist][var] += np.histogramdd([only_valid(d[mask], bins[hist][i][0]) for i,d in enumerate(data)], bins=bins[hist], weights=only_valid(w[mask] if w is not None else None))[0]
                except Exception as e:
                    warnings.warn(f'error filling {basename}/{hist} : '+str(e))

        # Return histograms
        return index, hists.copy(), event_list.copy()
    except Exception as e:
        print('Error:',filepath,e)
        return index, None, None


def save(f, filepath, hists, histograms, event_list, compression):
    filename = os.path.basename(filepath)

    compression_args = dict()
    if compression > 0:
        compression_args['compression'] = 'gzip'
        compression_args['compression_opts'] = compression

    # maybe create new event list
    if event_list is not None:
        if 'events' not in f:
            f.create_group('events')

        for filt in event_list:
            if filt not in f['events']:
                f['events'].create_group(filt)

            if filename not in f['events'][filt]:
                f['events'][filt].create_dataset(filename, data=np.array(event_list[filt]), **compression_args)

    for hist_name in hists:
        hist_config = histograms[hist_name]
        # maybe create new histogram entry
        if hist_name not in f:
            f.create_group(hist_name)
            f[hist_name].attrs['config'] = str(hist_config)
            for i,b in enumerate(hist_config['bins']):
                f[hist_name].attrs[f'bins{i}'] = generate_bins(b)

            f[hist_name].create_group('sum')
            f[hist_name].create_group('run')

        # maybe create new individual run dataset
        if filename not in f[hist_name]['run']:
            f[hist_name]['run'].create_group(filename)
            f[hist_name]['run'][filename].attrs['filepath'] = filepath

        # update run-level histograms
        hist_dict = hists[hist_name]
        run_grp = f[hist_name]['run'][filename]
        for hist in hist_dict:
            if hist not in run_grp:
                run_grp.create_dataset(hist, data=hist_dict[hist], **compression_args)
            else:
                run_grp[hist][:] = run_grp[hist][:] + hist_dict[hist]

        # update global histograms
        sum_grp = f[hist_name]['sum']
        for hist in hist_dict:
            if hist not in sum_grp:
                sum_grp.create_dataset(hist, data=hist_dict[hist], **compression_args)
            else:
                sum_grp[hist][:] = sum_grp[hist][:] + hist_dict[hist]

finished = []
def pool_callback(result):
    global finished
    finished.append(result)

def pool_error_callback(error):
    print(error)

def main(config_yaml, outpath, filepaths, compression, processes=None, batch_size=None, event_list=None, **kwargs):
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
        if 'field' not in hist_config:
            hist_config['field'] = None
        if 'weight' not in hist_config:
            hist_config['weight'] = None


    with h5py.File(outpath, 'a') as f:
        f['/'].attrs['config'] = str(config)

        processes = processes if processes is not None else multiprocessing.cpu_count()
        processes = min(processes, len(filepaths))

        print(f'Running on {processes} processes...')
        global finished
        with multiprocessing.Pool(processes) as p:
            results = {}
            pbar = tqdm.tqdm(enumerate(filepaths), smoothing=0, total=len(filepaths))
            for i,filepath in pbar:
                results[i] = p.apply_async(generate_histograms,
                                           tuple(),
                                           dict(
                                               index=i,
                                               filepath=filepath,
                                               histograms=config['histograms'],
                                               datasets=config['datasets'],
                                               variables=config.get('variables'),
                                               batch_size=batch_size,
                                               create_event_list=event_list is not None,
                                               imports=config.get('import')),
                                           callback=pool_callback, error_callback=pool_error_callback)

                while len(results) >= processes or ((i == len(filepaths)-1) and (len(results) > 0)):
                    while len(finished):
                        ifinish,hists,events = finished[0]
                        if hists is not None:
                            filepath = filepaths[ifinish]
                            pbar.desc = os.path.basename(filepath)
                            save(f, filepath, hists, config['histograms'], event_list=None if event_list is None else events, compression=compression)
                        del results[ifinish]
                        del finished[0]


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
    parser.add_argument('--compression', type=int, default=1,
                        help='''level of compression to apply to output''')
    parser.add_argument('--event_list', '-e', action='store_true', default=None,
                        help='''dump a list of events matching each filter to the file''')
    args = parser.parse_args()

    main(**vars(args))
