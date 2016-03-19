#!/usr/bin/env python
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--limit', type=int, default=50000)
    args = parser.parse_args()
    root = args.root.rstrip('/')
    print('''\
set term png size 6000,2000
set output '{root}/scores.png'
plot '{root}/scores' every ::::{limit} binary format='%1float32' using 1 with linespoints
    '''.format(root=root, limit=args.limit))
    for i in range(1, 37):
        print('''\
set output '{root}/state_{i}.png'
plot '{root}/states' every ::::{limit} binary format='%36float32' using {i} with linespoints
        '''.format(root=root, limit=args.limit, i=i))


if __name__ == '__main__':
    main()
