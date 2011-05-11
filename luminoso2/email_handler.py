import os
from dateutil.parser import parse as dateparse
from luminoso2.document_handlers import get_file_text

def handle_enron_file(url, name=None):
    email = [x.strip() for x in get_file_text(url).split('\n')]
    header = []
    body = []
    prev = []
    tags = []
    state = 'header'
    for line in email:
        if state == 'header':
            if line == '': 
                state = "body"
                continue
            header.append(line)
        if state == 'body':
            if line[:5] == '-----':
                state = "prev"
                continue
            body.append(line)
        if state == 'prev':
            prev.append(line)
    last = ''
    for line in header:
        part = line.split(':')[0]
        if part == 'Date':
            date = ":".join(line.split(':')[1:]).strip()
            datep = dateparse(date)
            dfinal = datep.date().isoformat()
            tags.append(('date', dfinal))
            last = 'Date'
            continue
        if part == 'From':
            addr = line.split(':')[1].strip().split('@')[0]
            tags.append(('from', addr.strip()))
            last = 'From'
            continue
        if part == 'To' or (last == 'To' and len(part) > 10):
            if last != 'To':
                addrs = [x.split('@')[0] for x in line.split(':')[1].strip().split(',') if x.strip() != '']
            else:
                addrsparts = line.strip().split(',')
                addrsparts = [x for x in addrsparts if len(x.split('@')) > 1]
                addrs = [x.split('@')[0] for x in addrsparts if x.split('@')[1] == 'enron.com']
            for addr in addrs:
                tags.append(('to', addr.strip()))
            last = 'To'
            continue
        last = 'Something'
    return {u'name': name, u'url': 'enron://'+url, u'text': ' '.join(body), u'tags': tags}


def handle_enron(url):
    if url.startswith('enron://'): url = url[8:]
    if not isinstance(url, unicode): url = url.decode('utf-8')
    
    # Handle a single file
    if not os.path.isdir(url):
        yield handle_enron_file(url)
        return

    # Handle a directory.
    for user in os.listdir(url):
        user_path = os.path.join(url, user)
        if user.startswith('.') or not os.path.isdir(user_path): continue
        for path, dirs, files in os.walk(user_path):
            relpath = path.replace(url, u'').lstrip(os.sep)
            if 'discussion_threads' in dirs:
                dirs.remove('discussion_threads')
            for filename in files:
                if filename.startswith('.'): continue
                yield handle_enron_file(os.path.join(path, filename), name='%s/%s' % (relpath, filename))

