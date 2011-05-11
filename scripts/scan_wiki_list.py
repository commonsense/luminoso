#!/usr/bin/env python
"""
scan_wiki_list.py -- reads a Wikipedia "Featured Articles" or "Good Articles"
list, separates it into categories at `<h3>` headings, and outputs a .json file
for each category.
"""

from BeautifulSoup import BeautifulSoup
from urllib2 import urlopen, Request
from collections import defaultdict
import json

SPOOF_HEADERS = {'User-Agent': "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11"}

def scan_wiki_list(urls):
    section = None
    found = defaultdict(list)
    for url in urls:
        domain = '/'.join(url.split('/')[:3])
        req = Request(url, headers=SPOOF_HEADERS)
        for line in urlopen(req):
            if line.startswith('<h3>'):
                soup = BeautifulSoup(line.decode('utf-8'))
                section = soup.find(attrs={'class': 'mw-headline'}).renderContents()
                print section
            elif line.startswith('<li>') or line.startswith('<p>') and section is not None:
                soup = BeautifulSoup(line.decode('utf-8'))
                for link in soup.findAll(name='a'):
                    if link['href'].startswith('/'):
                        target = domain + link['href']
                        name = link.renderContents()
                        info = {
                            'url': target,
                            'name': name,
                        }
                        if ':' not in name:
                            found[section].append(info)
    assert found
    for key, pages in found.items():
        if key and len(pages) >= 5:
            filename = key+'.json'
            out = open(filename, 'w')
            json.dump(pages, out)
            out.close()
