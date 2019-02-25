import argparse
from NN.b import process

import cv2

img = None

# class StartAction(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         print("Hello")

parser = argparse.ArgumentParser(description='Discern photos.')
parser.add_argument('files', nargs=2)
parser.add_argument('--out', nargs='?')
args = parser.parse_args()

result = process(args.files[0], args.files[1], args.out)
if args.out:
    cv2.imwrite(args.out, result)

# parser.add_argument('-s', '--start', action=StartAction, nargs=0)
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
