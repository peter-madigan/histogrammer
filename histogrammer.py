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

    constants:
      <constant name>:
        path: <path to object within input hdf5 file>
        name: <field name or attribute name to load from "path">
        attr: <bool, true if constant is to be fetched from group/dataset attribute rather than attempting to load a dataset>
        expr: <str, a python expression to evaluate once instead of loading from a file, other constants defined in namespace can be used locally, optional>

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
        if 'log' in spec and spec['log'] == True:
            bins = np.geomspace(spec['low'], spec['high'], spec['n']+1)
        else:
            bins = np.linspace(spec['low'], spec['high'], spec['n']+1)
    else:
        bins = np.array(spec)
    return np.sort(bins)


def only_valid(maybe_masked_arr, fill=0):
    if maybe_masked_arr is None:
        return None
    return (maybe_masked_arr.ravel()
            if not ma.is_masked(maybe_masked_arr)
            else maybe_masked_arr.filled(fill))


def generate_histograms(lock, index, input_filepath, output_filepath, histograms, *args, datasets=None, variables=None, constants=None, batch_size=None, imports=None, event_list_variables=None, verbose=True, runxrun=False, compression=0, **kwargs):
    then = time.time()
    now = time.time()

    if imports is not None:
        for lib in imports:
            globals()[lib] = __import__(lib)

    # Open the file
    try:
        f = H5FlowDataManager(input_filepath, 'r', mpi=False)
        basename = os.path.basename(input_filepath)

        # Initialize histogram(s)
        bins = dict()
        hists = dict()
        for hist in histograms:
            bins[hist] = [generate_bins(b) for b in histograms[hist]['bins']]
            hists[hist] = dict()
            hists[hist]['overall'] = np.zeros([len(b)-1 for b in bins[hist]])
            if verbose:
                print(hist, histograms[hist])
            if variables:
                for var in variables:
                    if variables[var].get('filt', True):
                        hists[hist][var] = np.zeros([len(b)-1 for b in bins[hist]])

        # Initialize loop
        loop_dataset = [datasets[d] for d in datasets if 'loop' in datasets[d] and datasets[d]['loop']][0]
        rows = f[loop_dataset['path'][0] + '/data'].shape[0]
        if batch_size:
            batches = [slice(i,min(i+batch_size,rows)) for i in range(0,rows,batch_size)]
        else:
            batches = [slice(0,rows)]

        # Initialize constant values
        if constants:
            for c_name, c_config in constants.items():
                path = c_config.get('path', None)
                name = c_config.get('name', None)
                attr = c_config.get('attr', False)
                expr = c_config.get('expr', None)

                if attr == True:
                    const = f[path].attrs[name]
                else:
                    if expr is not None:
                        const = eval(expr)
                    elif path is not None:
                        const = f[path][:] if name is None else f[path][name]
                    else:
                        raise RuntimeError(f'Could not parse constant {c_name}: {c_config}')

                globals()[c_name] = const

                if verbose:
                    print(c_name, const)
                    
        # Initialize filter operations
        var_op = dict()
        if variables:
            for var in variables:
                expr = variables[var]['expr']
                if verbose:
                    print(var, expr)
                var_op[var] = eval('lambda : ' + expr)

        # Run loop
        event_list = dict()
        if variables and event_list_variables:
            for var in event_list_variables:
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
                        if verbose:
                            print(dset, globals()[dset].shape)
                    except Exception as e:
                        if verbose:
                            warnings.warn(f'error in {basename}/{dset} : '+str(e))
                        globals()[dset] = None

            # Load variables
            if variables:
                for var in variables:
                    # Apply variable function
                    try:
                        globals()[var] = var_op[var]()
                        if event_list_variables and variables[var].get('filt', True) and var in event_list_variables:
                            event_list[var] += (np.where(globals()[var])[0] + batch.start).tolist()
                        if verbose:
                            print(var, globals()[var].shape, globals()[var].min(), globals()[var].max())
                    except Exception as e:
                        if verbose:
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
                            try:
                                if variables[var].get('filt', True) and np.any(globals()[var]):
                                    mask = globals()[var].astype(bool)
                                    hists[hist][var] += np.histogramdd([only_valid(d[mask], bins[hist][i][0]) for i,d in enumerate(data)], bins=bins[hist], weights=only_valid(w[mask] if w is not None else None))[0]
                            except Exception as e:
                                if verbose:
                                    warnings.warn(f'error filling {basename}/{hist}/{var} : '+str(e))
                except Exception as e:
                    if verbose:
                        warnings.warn(f'error filling {basename}/{hist} : '+str(e))

        # Return histograms
        now = time.time()
        print('loop on:', now-then)
        then = now

        lock.acquire()
        try:
            print(input_filepath, event_list)
            save(
                output_filepath,
                input_filepath,
                hists,
                histograms,
                event_list=None if event_list is None else event_list,
                compression=compression,
                runxrun=runxrun,
                verbose=verbose)
        except Exception as e:
            if verbose:
                print('Error:', input_filepath, e)
        finally:
            lock.release()

        #return index, hists.copy(), event_list.copy()
        return index, None, None
    except Exception as e:
        if verbose:
            print('Error:', input_filepath, e)
        return index, None, None

save_cache = dict()
def save(outpath, filepath, hists, histograms, event_list, compression, runxrun=None, flush_cache=True, verbose=False):
    then = time.time()
    now = time.time()
    
    filename = os.path.basename(filepath)
    if verbose:
        print(f'Saving {filename}')

    with h5py.File(outpath, 'a') as f:

        compression_args = dict()
        if compression > 0:
            compression_args['compression'] = 'gzip'
            compression_args['compression_opts'] = compression
            if verbose:
                print(f'With compression: {compression_args}')

        # maybe create new event list
        if event_list is not None:
            if 'events' not in f:
                f.create_group('events')

            for filt in event_list:
                if filt not in f['events']:
                    f['events'].create_group(filt)

                if filename not in f['events'][filt]:
                    if verbose:
                        print(f'\t{filename}: {len(event_list[filt])} events in event list')
                    f['events'][filt].create_dataset(filename, data=np.array(event_list[filt]), **compression_args)

        for hist_name in hists:
            hist_config = histograms[hist_name]
            # maybe create new histogram entry
            if hist_name not in f:
                if verbose:
                    print(f'\tCreating new histogram {hist_name}')
            
                f.create_group(hist_name)
                f[hist_name].attrs['config'] = str(hist_config)
                for i,b in enumerate(hist_config['bins']):
                    f[hist_name].attrs[f'bins{i}'] = generate_bins(b)

                f[hist_name].create_group('sum')
                f[hist_name].create_group('run')

            hist_dict = hists[hist_name]

            # update global histograms
            sum_grp = f[hist_name]['sum']
            for hist in hist_dict:
                if hist not in sum_grp:
                    sum_grp.create_dataset(hist, data=hist_dict[hist], **compression_args)
                else:
                    if flush_cache:
                        if np.any(save_cache.get(hist_name + '/' + hist, 0) + hist_dict[hist] != 0):
                            sum_grp[hist][:] = sum_grp[hist][:] + hist_dict[hist] + save_cache.get(hist_name + '/' + hist, 0)
                    else:
                        save_cache[hist_name + '/' + hist] = save_cache.get(hist_name + '/' + hist, 0) + hist_dict[hist]
                if verbose:
                    print(f'\tAdded {hist_dict[hist].sum()} new entries to {hist_name}/sum/{hist}')

            if runxrun is not None:
                # maybe create new individual run dataset
                if filename not in f[hist_name]['run']:
                    f[hist_name]['run'].create_group(filename)
                    f[hist_name]['run'][filename].attrs['filepath'] = filepath

                # update run-level histograms
                run_grp = f[hist_name]['run'][filename]
                for hist in hist_dict:
                    if hist not in run_grp:
                        run_grp.create_dataset(hist, data=hist_dict[hist], **compression_args)
                    else:
                        run_grp[hist][:] = run_grp[hist][:] + hist_dict[hist]

    now = time.time()
    print('save:', now-then)
    then = now

finished = []
def pool_callback(result):
    global finished
    finished.append(result)

def pool_error_callback(error):
    print(error)

def main(config_yaml, outpath, filepaths, compression, processes=None, batch_size=None, event_list=None, update=None, verbose=False, runxrun=None, **kwargs):
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

        if update is not None:
            try:
                with h5py.File(outpath, 'a') as f:
                    if hist_name in f and hist_name not in update:
                        # histogram exists and we don't explicitly want to update it
                        if verbose:
                            print(f'Skipping existing histogram {hist_name}')
                        del config['histograms'][hist_name]
                    elif hist_name in f and hist_name in update:
                        # histogram exists, and we do want to update it, so lets reset it
                        print(f'Regenerating existing histogram {hist_name}')
                        del f[hist_name]
                    else:
                        # histogram doesn't exist, so just do nothing
                        pass
                        
            except Exception as e:
                print(f'Error occurred when trying to check if histogram {hist_name} exists: {e}')

    with h5py.File(outpath, 'a') as f:
        f['/'].attrs['config'] = str(config)

    processes = processes if processes is not None else multiprocessing.cpu_count()
    processes = min(processes, len(filepaths))

    print(f'Running on {processes} processes...')
    global finished
    with multiprocessing.Manager() as mgr:
        lock = mgr.Lock()
        
        with multiprocessing.Pool(processes) as p:
            results = {}
            pbar = tqdm.tqdm(enumerate(filepaths), smoothing=0, total=len(filepaths))
            i = 0
            while True:
                #for i,filepath in pbar:
                if i < len(filepaths):
                    filepath = filepaths[i]
                    results[i] = p.apply_async(generate_histograms,
                                           tuple(),
                                           dict(
                                               lock=lock,
                                               index=i,
                                               input_filepath=filepath,
                                               output_filepath=outpath,
                                               histograms=config['histograms'],
                                               datasets=config['datasets'],
                                               variables=config.get('variables'),
                                               constants=config.get('constants'),
                                               batch_size=batch_size,
                                               event_list_variables=event_list,
                                               verbose=verbose,
                                               runxrun=runxrun,
                                               compression=compression,
                                               imports=config.get('import')),
                                           callback=pool_callback, error_callback=pool_error_callback)
                    i += 1

                while len(results) >= processes or ((i == len(filepaths)) and (len(results) > 0)):
                    if len(finished):
                        ifinish,hists,events = finished[0]
                        if hists is not None:
                            filepath = filepaths[ifinish]
                            pbar.desc = os.path.basename(filepath)
                            save(output_filepath, filepath, hists, config['histograms'], event_list=None if event_list is None else events, compression=compression, runxrun=runxrun, flush_cache=len(results)==1, verbose=verbose)
                        del results[ifinish]
                        del finished[0]
                        pbar.update(1)

                if (i == len(filepaths)) and (len(results) == 0):
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outpath', '-o', type=str,
        required=True, help='''output hdf5 file to generate''')
    parser.add_argument('--filepaths', '-i', nargs='+', type=str, required=True,
        help='''input h5flow files to generate histograms from''')
    parser.add_argument('--config_yaml', '-c', type=str, required=True,
        help='''yaml config file''')
    parser.add_argument('--update', nargs='+', type=str,
                        help='''only update the specified histograms''')
    parser.add_argument('--processes', '-p', type=int, default=None,
        help='''number of parallel processes (defaults to number of cpus detected)''')
    parser.add_argument('--batch_size', '-b', type=int, default=None,
        help='''batch size for loop (optional)''')
    parser.add_argument('--compression', type=int, default=0,
                        help='''level of compression to apply to output''')
    parser.add_argument('--event_list', '-e', nargs='+', default=None,
                        help='''dump a list of events matching each filter to the file''')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='''create more output''')
    parser.add_argument('--runxrun', action='store_true', default=None,
                        help='''save histograms run-by-run in file''')
    args = parser.parse_args()

    main(**vars(args))
