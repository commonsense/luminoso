#!/usr/bin/env python
from luminoso import LuminosoModel
model = LuminosoModel.make_common_sense('/tmp/luminoso_profile', 'en')

def run():
    model.add_from_url('../models/PLDBStudy/Documents')

if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run('run()', '/tmp/luminoso_profile/add_docs.profile')
    p = pstats.Stats('/tmp/luminoso_profile/add_docs.profile')
    p.sort_stats('time').print_stats(100)

