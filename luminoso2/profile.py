#!/usr/bin/env python
from luminoso2 import LuminosoModel
import os, shutil
if os.access('/tmp/luminoso_profile', os.F_OK):
    shutil.rmtree('/tmp/luminoso_profile')
model = LuminosoModel.make_common_sense('/tmp/luminoso_profile', 'ja')

def run():
    model.add_from_url('../models/ja_tweets')

if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run('run()', '/tmp/luminoso_profile/add_docs.profile')
    p = pstats.Stats('/tmp/luminoso_profile/add_docs.profile')
    p.sort_stats('cum').print_stats(100)

