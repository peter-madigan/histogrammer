import setuptools

setuptools.setup(name='histogrammer',
                 version=version,
                 description='A single script tool for generating histograms from an h5flow-based analysis',
                 long_description='',
                 author='Peter Madigan',
                 author_email='pmadigan@berkeley.edu',
                 packages=setuptools.find_packages(where='.'),
                 python_requires='>=3.7',
                 install_requires=[
                     'h5py>=2.10',
                     'h5flow>=0.1.0',
                     'numpy',
                     'tqdm',
                     'pyyaml'
                 ]
                 )
