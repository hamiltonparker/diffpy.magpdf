#!/usr/bin/env python

# Installation script for diffpy.mpdf

"""mpdf - Tools for magnetic pair distribution function analysis.
Packages:   diffpy.mpdf
"""

import os
from setuptools import setup, find_packages

# Use this version when git data are not available, like in git zip archive.
# Update when tagging a new release.
FALLBACK_VERSION = '1.0.1.post0'

# versioncfgfile holds version data for git commit hash and date.
# It must reside in the same directory as version.py.
MYDIR = os.path.dirname(os.path.abspath(__file__))
versioncfgfile = os.path.join(MYDIR, 'diffpy/mpdf/version.cfg')
gitarchivecfgfile = versioncfgfile.replace('version.cfg', 'gitarchive.cfg')

def gitinfo():
    from subprocess import Popen, PIPE
    kw = dict(stdout=PIPE, cwd=MYDIR)
    proc = Popen(['git', 'describe', '--match=v[[:digit:]]*'], **kw)
    desc = proc.stdout.read()
    proc = Popen(['git', 'log', '-1', '--format=%H %at %ai'], **kw)
    glog = proc.stdout.read()
    rv = {}
    rv['version'] = '.post'.join(desc.strip().split('-')[:2]).lstrip('v')
    rv['commit'], rv['timestamp'], rv['date'] = glog.strip().split(None, 2)
    return rv


def getversioncfg():
    import re
    try:
        from configparser import RawConfigParser
    except:
        from ConfigParser import RawConfigParser
    vd0 = dict(version=FALLBACK_VERSION, commit='', date='', timestamp=0)
    # first fetch data from gitarchivecfgfile, ignore if it is unexpanded
    g = vd0.copy()
    cp0 = RawConfigParser(vd0)
    cp0.read(gitarchivecfgfile)
    if '$Format:' not in cp0.get('DEFAULT', 'commit'):
        g = cp0.defaults()
        mx = re.search(r'\btag: v(\d[^,]*)', g.pop('refnames'))
        if mx:
            g['version'] = mx.group(1)
    # then try to obtain version data from git.
    gitdir = os.path.join(MYDIR, '.git')
    if os.path.exists(gitdir) or 'GIT_DIR' in os.environ:
        try:
            g = gitinfo()
        except OSError:
            pass
    # finally, check and update the active version file
    cp = RawConfigParser()
    cp.read(versioncfgfile)
    d = cp.defaults()
    rewrite = not d or (g['commit'] and (
        g['version'] != d.get('version') or g['commit'] != d.get('commit')))
    if rewrite:
        cp.set('DEFAULT', 'version', g['version'])
        cp.set('DEFAULT', 'commit', g['commit'])
        cp.set('DEFAULT', 'date', g['date'])
        cp.set('DEFAULT', 'timestamp', g['timestamp'])
        cp.write(open(versioncfgfile, 'w'))
    return cp

#versiondata = getversioncfg()  ### python3 incompatible

# define distribution
setup_args = dict(
        name = "diffpy.mpdf",
        #version = versiondata.get('DEFAULT', 'version'),
        version = '1.0.0',
        namespace_packages = ['diffpy'],
        packages = find_packages(),
        test_suite = 'diffpy.mpdf.tests',
        include_package_data = True,
        zip_safe = False,
        author = 'Simon J.L. Billinge group',
        author_email = 'sb2896@columbia.edu',
        maintainer = 'Benjamin Frandsen',
        maintainer_email = 'benfrandsen@gmail.com',
        description = "Tools for magnetic PDF analysis.",
        license = 'BSD-style license',
        url = "https://github.com/benfrandsen/diffpy.mpdf/",
        keywords = "mPDF magnetic PDF",
        classifiers = [
            # List of possible values at
            # http://pypi.python.org/pypi?:action=list_classifiers
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Programming Language :: Python :: 2.5',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering :: Physics',
        ],
)

if __name__ == '__main__':
    setup(**setup_args)

# End of file
