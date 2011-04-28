import os, codecs
from dateutil.parser import parse as dateparse

def handle_enron_file(url, name=None):
    mailfile = codecs.open(url, 'r', encoding='utf-8')
    email = [x.strip() for x in mailfile.readlines()]
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
    return {u'name': name, u'url': url, u'text': ' '.join(body), u'tags': tags}


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
        for folder in os.listdir(user_path):
            folder_path = os.path.join(user_path, folder)
            if folder.startswith('.') or not os.path.isdir(folder_path): continue
            if folder == 'discussion_threads': continue
            for message in os.listdir(folder_path):
                if message.startswith('.'): continue
                yield handle_enron_file(os.path.join(folder_path, message), name='%s/%s/%s' % (user, folder, message))

