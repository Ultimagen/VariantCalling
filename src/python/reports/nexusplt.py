import os
import matplotlib.pyplot as plt

try:
    import mpld3
except ImportError as error:
    print("WARNING: mpld3 not available")
    MPLD3 = False
    pass
else:
    MPLD3 = True

# supported file formats
EXTS = ['png', 'html', 'json']


def fname(fname, outdir=''):
    '''
    generates full filepath; creates directory when necessary

    fname  : filename
    outdir : output directory
             DEFAULT: current working directory
    '''
    if not outdir:
        outdir = os.getcwd()
    elif not os.path.exists(outdir):
        try:
            os.mkdir(outdir)
        except:
            print(f"Could not create {outdir} directory")

    if os.path.isdir(outdir):
        return os.path.join(outdir, fname)
    else:
        print(f"{outdir} is somehow not a directory")


def save(fig, fspec, ext='', outdir='', verbose=False):
    '''
    saves a matplotlib Figure in either PNG, JPG or HTML format

    fig    : matplotlib.figure.Figure
    fspec  : either filename with extension, or file basename
    ext    : filename extension (applies when `fspec` is a file basename)
             DEFAULT: 'png'
    outdir : output directory (applies when `fpsec` is a file basename)
    '''

    fn = ''
    if any(fspec.endswith(f".{x}") for x in EXTS):
        fn = fspec
    else:
        if any(ext is x for x in EXTS):
            fn = fname(f"{fspec}.{ext}", outdir=outdir)
        else:
            ext = ext or fspec.split('.')[-1]
            print(f"Unrecognized extension: {ext}")

    if verbose:
        print(fn)

    if fn.endswith('.png'):
        plt.savefig(fn);
    elif fn.endswith('.json') or fn.endswith('.html'):
        if MPLD3:
            try:
                if '.json' in fn:
                    mpld3.save_json(fig, fn)
                else:
                    mpld3.save_html(fig, fn)
            except Exception as ex:
                print(f"Cannot save {fn}")
                print(repr(ex))
                pass


def save_all(fig, fbase, outdir='', verbose=False):
    '''
    saves a matplotlib Figure in 3 formats (PNG, JPG and HTML)

    fig    : matplotlib.figure.Figure
    fbase  : file basename (without extension)
    outdir : output directory

    '''
    for ext in EXTS:
        save(fig, fbase, ext=ext, outdir=outdir, verbose=verbose)
