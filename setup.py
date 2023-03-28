try:
    from setuptools import setup
except (ImportError,ModuleNotFoundError):
    from distutils.core import setup
import glob
import os
import sys

def package_files(package_dir,subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths
#data_files = package_files('src','data')

setup_args = {
    'name': 'sdrudp',
    'author': 'UC Berkeley RadioLab',
    'url': 'https://github.com/AaronParsons/sdrudp',
    'description': 'Tools for managing SDRs across networks.',
    'package_dir': {'sdrudp': 'src'},
    'packages': ['sdrupd'],
    'include_package_data': True,
    'scripts': glob.glob('scripts/*'),
    'version': '0.0.1',
    #'package_data': {'sdrudp': data_files},
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
