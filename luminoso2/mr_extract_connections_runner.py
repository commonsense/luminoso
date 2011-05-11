#!/usr/bin/env python
from mr_extract_connections import ConnectionExtractor
import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    out = args.pop(0)

    job = ConnectionExtractor(args=args+['--no-output'])
    with open(out, 'wb') as output:
        with job.make_runner() as runner:
            runner.run()
            for line in runner.stream_output():
                output.write(line)

