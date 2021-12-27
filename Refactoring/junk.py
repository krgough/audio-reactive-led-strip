#!/usr/bin/env python3

from getopt import gnu_getopt, GetoptError
import os
import sys
from textwrap import dedent


def spectrum():
    print('spectrum')


def scroll():
    print('scroll')


def energy():
    print('energy')


def get_args():
    """ Get command line arguments and return the selected visualisation function
    """
    usage = dedent(f"""
    USAGE: ./{os.path.basename(__file__)} [-h] spectrum|energy|scroll

    Generates visual LED display with selected visualisation function
    """)

    try:
        opts, args = gnu_getopt(sys.argv[1:], "h")
        for opt, _ in opts:
            if opt == '-h':
                print(usage)
                sys.exit(0)

    except GetoptError:
        print("ERROR: Invalid argument")
        print(usage)
        sys.exit(1)

    if args:
        viz_func = args[0]
        funcs = {'spectrum': spectrum, 'scroll': scroll, 'energy': energy}    
        if viz_func in funcs:
            return funcs[viz_func]
        else:
            print(f'ERROR: Visualisation function {viz_func} is not recognised')
            sys.exit(1)
    else:
        print("ERROR: No visualisation function given")
        print(usage)
        sys.exit(1)



func = get_args()
func()
print('all done')

