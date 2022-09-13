import seaborn as sns


class pcolors:
    """Helper function to print colored output.

    Example: print(bcolors.WARNING + "Warning" + bcolors.ENDC)
    """
    def __init__(self, palette = sns.color_palette("deep")):
        self.palette = palette
        self.train = palette[1]
        self.val = palette[3]
        self.lr = palette[5]
        self.test = palette[4]
        self.fun = palette[0]
    # plt.rcParams['text.usetex'] = True
