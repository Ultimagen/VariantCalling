import pandas as pd

if __name__ == '__main__':
    bl = pd.read_hdf("blacklist_ua_good_old_blacklist.h5")
    bl = bl.reset_index()['index']
    chrom = bl.apply(lambda x: x[0])
    pos = bl.apply(lambda x: x[1])
    df = pd.DataFrame({'chrom': chrom, 'start': pos - 1, 'end': pos})
    df.to_csv('blacklist.bed', sep='\t', index=False)
