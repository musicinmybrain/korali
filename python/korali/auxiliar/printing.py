#! /usr/bin/env python3

class bcolors:
    """ Background Colors
    Helper function to print colored output.

    Example: print(bcolors.WARNING + "Warning" + bcolors.ENDC)
    """

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m' # End colored output
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text="", width=80, sep="=", color=None):
    """Print header with seperator.

    :param text:
    :param width:
    :param sep:
    :returns:
    """
    if len(text) == 0:
        text = sep*width
        if color:
            text = color + text + bcolors.ENDC
        print(text)
    else:
        txt_legnth = len(text)+2
        fill_width = int((width-txt_legnth)/2)
        if color:
            text = color + text + bcolors.ENDC
        print(sep*fill_width+" "+text+" "+sep*fill_width)


def print_args(d, header_text="Running with args", color=None, width=30, header_width=80, sep="="):
    """Print args from args parser formated nicely.

    :param d: dictonary of args
    :param heder_text:
    :param width:
    :param header_width:
    :param sep:
    :returns:
    """
    print_header(header_text, color=color, sep=sep)
    for key, value in d.items():
        if value:
            out_string = '\t{:<{width}} {:<}'.format(key, value, width=width)
            print(out_string)
    print_header(sep=sep, width=header_width)
