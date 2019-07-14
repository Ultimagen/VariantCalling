import pandas as pd
import numpy as np
ERROR_PROBS = "/home/ilya/proj/VariantCalling/work/190628/probability.csv"


def read_error_probs(error_probs_csv: str = ERROR_PROBS) -> pd.DataFrame:
    '''Read error probs CSV and produces data frame

    Parameters
    ----------
    error_probs_csv: str
            CSV file with error probabilities. Example %s

    Returns
    -------
    pd.DataFrame
            DataFrame with multiindex: left context, hmer nuc, hmer length, right context
            and columns:
    ''' % ERROR_PROBS

    source_dataframe = pd.read_csv(error_probs_csv)
    source_dataframe['left'] = source_dataframe['motif'].apply(lambda x: x[:5])
    source_dataframe['right'] = source_dataframe['motif'].apply(lambda x: x[-5:])
    source_dataframe['middle'] = source_dataframe['motif'].apply(lambda x: x[6:8])
    source_dataframe['hmer_letter'] = source_dataframe['middle'].apply(lambda x: x[1])
    source_dataframe['hmer_number'] = source_dataframe['middle'].apply(lambda x: x[0]).astype(np.int)
    source_dataframe.drop(['motif', 'middle'], axis=1, inplace=True)
    tups = [tuple(x) for x in source_dataframe[['left', 'hmer_number', 'hmer_letter', 'right']].values]
    source_dataframe.index = pd.MultiIndex.from_tuples(tups, names=['left', 'hmer_number', 'hmer_letter', 'right'])

    source_dataframe.drop(['left', 'right', 'hmer_letter', 'hmer_number'], axis=1, inplace=True)

    source_dataframe.columns = ['n(-1)', 'n(0)', 'n(+1)', 'P(-1)', 'P(0)', 'P(+1)']
    return source_dataframe


def _convert_to_probs(source_dataframe: pd.DataFrame):
    '''Converts counts to probabilities

    Parameters
    ----------
    source_dataframe: pd.DataFrame
            DataFrame of counts

    Returns
    -------
    adds columns P(-1), P(0), P(+1) to convert counts (n(-1), n(0),n(+1)) to probabilities

    '''

    sum_counts = source_dataframe[['n(-1)', 'n(0)', 'n(+1)']].sum(axis=1)
    probs = source_dataframe[['n(-1)', 'n(0)', 'n(+1)']].multiply(1 / sum_counts, axis=0)

    probs.columns = ['P(-1)', 'P(0)', 'P(+1)']
    source_dataframe[['P(-1)', 'P(0)', 'P(+1)']] = probs
    return source_dataframe


def marginalize_error_probs(source_dataframe: pd.DataFrame, left_drop: int = 0, right_drop: int=0) -> pd.DataFrame:
    '''Marginalize error probabilities by combining motifs sharing common suffix (left) or prefix (right)
    This function is useful for calculation of error probabilities of nucleotides that are close to the end 
    of the read

    Parameters
    ----------
    source_dataframe: pd.DataFrame
            Input DataFrame
    left_drop: int
            Number of nucleotides to marginalize on in prefix of left context
    right_drop: int
            Number of nucleotides to marginalize on in suffix of right context
    '''
    source_dataframe = source_dataframe.drop(0, axis=0, level='hmer_number').copy()
    assert left_drop > 0 or right_drop > 0, "No marginalization needed for these drop values"
    len_left = len(source_dataframe.index.get_level_values("left")[0])
    len_right = len(source_dataframe.index.get_level_values("right")[0])
    assert left_drop <= len_left and right_drop <= len_right, "Unable to marginalize on more nucs than exist"

    groupby_left = source_dataframe.index.get_level_values('left').str[left_drop:]
    if right_drop > 0:

        groupby_right = source_dataframe.index.get_level_values('right').str[:-right_drop]
    else:
        groupby_right = source_dataframe.index.get_level_values('right')
    groupby_length = source_dataframe.index.get_level_values('hmer_number')
    groupby_hmer = source_dataframe.index.get_level_values('hmer_letter')

    mi = pd.MultiIndex.from_arrays((groupby_left, groupby_right, groupby_length, groupby_hmer))
    source_dataframe1 = source_dataframe[['n(-1)', 'n(0)', 'n(+1)']].copy()
    source_dataframe1.index = mi
    gsource_dataframe = source_dataframe1.groupby(axis=0, level=[0, 1, 2, 3])
    source_dataframe1 = gsource_dataframe.agg(np.sum)
    source_dataframe1 = _convert_to_probs(source_dataframe1)
    source_dataframe = source_dataframe1.reorder_levels(['left', 'hmer_number', 'hmer_letter', 'right'], axis=0)
    source_dataframe = add_zero_model(source_dataframe)
    source_dataframe.sort_index(inplace=True)
    return source_dataframe


def create_marginalize_dictionary(source_dataframe: pd.DataFrame) -> dict:
    '''Creates a dictionary of all possible marginalizations of the error model

    Parameters
    ----------
    source_dataframe: pd.DataFrame
        original dataframe

    Returns
    -------
    dict: 
        dictionary with keys - sizes of the left and right motif
    '''

    marginalize_dict = {}
    len_left = len(source_dataframe.index.get_level_values("left")[0])
    len_right = len(source_dataframe.index.get_level_values("right")[0])

    marginalize_dict[(len_left, len_right)] = source_dataframe.copy()
    for i in range(len_left + 1):
        for j in range(len_right + 1):
            if (len_left - i, len_right - j) in marginalize_dict:
                continue
            elif (len_left - i + 1, len_right - j) in marginalize_dict:
                source_dataframe = marginalize_dict[((len_left - i + 1, len_right - j))]
                source_dataframe1 = marginalize_error_probs(source_dataframe, 1, 0)
                marginalize_dict[(len_left - i, len_right - j)] = source_dataframe1
            elif (len_left - i, len_right - j + 1) in marginalize_dict:
                source_dataframe = marginalize_dict[((len_left - i, len_right - j + 1))]
                source_dataframe1 = marginalize_error_probs(source_dataframe, 0, 1)
                marginalize_dict[(len_left - i, len_right - j)] = source_dataframe1
            else:
                raise Exception("Can't create {}".format((len_left - i, len_right - j)))
    return marginalize_dict

def add_zero_model( source_dataframe: pd.DataFrame ) -> pd.DataFrame: 
    '''Add probabilities for 0->0 and 0->1 errors.  
    Due to implementation difficulties P(0|0) and P(0|1) were not reported.
    The calculation is P(0|1) = P (1|0), P(0|0) = 1-P(1|0)

    Parameters
    ----------
    source_dataframe: pd.DataFrame
        Input dataframe

    Return
    ------
    pd.DataFrame
    '''
    new_source_dataframe = source_dataframe.xs(1, axis=0, level='hmer_number').copy()
    new_source_dataframe = pd.concat((new_source_dataframe,),keys=[0], names=['hmer_number'])
    new_source_dataframe = new_source_dataframe.reorder_levels(['left', 'hmer_number', 'hmer_letter', 'right'], axis=0)

    # there is no meaning for counts for now in this case
    new_source_dataframe[['n(0)', 'n(-1)', 'n(+1)']] = 0 
    new_source_dataframe[['P(+1)']] = new_source_dataframe[['P(-1)']]
    new_source_dataframe[['P(-1)']] = 0 
    new_source_dataframe[['P(0)']] = 1-new_source_dataframe[['P(+1)']]
    return pd.concat((new_source_dataframe, source_dataframe))

def convert2readGivenData( source_dataframe: pd.DataFrame) -> pd.DataFrame : 
    '''Convert probabilities in P(read | data = i ) to P(read=i | data)

    Parameters
    ----------
    source_dataframe: pd.DataFrame
        source 

    Returns
    -------
    pd.DataFrame
    '''

    left_names = source_dataframe.index.get_level_values('left').unique()
    hmer_names = source_dataframe.index.get_level_values('hmer_letter').unique()
    right_names = source_dataframe.index.get_level_values('right').unique()
    hmer_number = source_dataframe.index.get_level_values('hmer_number').max()
    hmer_number = np.arange(0, hmer_number+2)

    idx = pd.MultiIndex.from_product([left_names, hmer_number,hmer_names,  right_names],
                                    names = ['left','hmer_number', 'hmer_letter','right']).sort_values()
    result_dataframe = pd.DataFrame(index=idx,columns=source_dataframe.columns)

    source_dataframe['dest(-1)'] = source_dataframe.index.get_level_values('hmer_number')+1
    source_dataframe['dest(0)'] = source_dataframe.index.get_level_values('hmer_number').astype(np.int)
    source_dataframe['dest(+1)'] = source_dataframe.index.get_level_values('hmer_number')-1

    # now rearranging 
    tmp = source_dataframe[source_dataframe['dest(-1)']>=0]
    tmp = tmp.reset_index()
    tmp.sort_values(['left', 'dest(-1)','hmer_letter', 'right'], inplace=True)
    dest_index = pd.MultiIndex.from_frame(tmp[['left', 'dest(-1)','hmer_letter', 'right']],
                                         names = ['left','hmer_number','hmer_letter','right'])

    result_dataframe.loc[dest_index,'n(-1)'] = np.array(tmp['n(+1)'].values)
    result_dataframe.loc[dest_index,'P(-1)'] = np.array(tmp['P(+1)'].values)

    tmp = source_dataframe[source_dataframe['dest(0)']>=0]
    tmp = tmp.reset_index()
    tmp.sort_values(['left', 'dest(-1)','hmer_letter', 'right'], inplace=True)

    dest_index = pd.MultiIndex.from_frame(tmp[['left', 'dest(0)','hmer_letter', 'right']],
                                         names = ['left','hmer_number','hmer_letter','right'])
    result_dataframe.loc[dest_index,'n(0)'] = np.array(tmp['n(0)'].values)
    result_dataframe.loc[dest_index,'P(0)'] = np.array(tmp['P(0)'].values)

    tmp = source_dataframe[source_dataframe['dest(+1)']>=0]
    tmp = tmp.reset_index()
    tmp.sort_values(['left', 'dest(-1)','hmer_letter', 'right'], inplace=True)

    dest_index = pd.MultiIndex.from_frame(tmp[['left', 'dest(+1)','hmer_letter', 'right']],
                                         names = ['left','hmer_number','hmer_letter','right'])
    result_dataframe.loc[dest_index,'n(+1)'] = np.array(tmp['n(-1)'].values)
    result_dataframe.loc[dest_index,'P(+1)'] = np.array(tmp['P(-1)'].values)
    source_dataframe.drop(['dest(-1)','dest(0)', 'dest(+1)'], axis=1, inplace=True)

    result_dataframe.dropna(how='all',inplace=True)

    return result_dataframe

class ErrorModel : 
    '''Contains error model and functions to access it. The design of the class 
    is mosty for efficiency purposes

    Attributes
    ----------
    _em - error model. Currently implemented as numpy array, keeps only probabilities
    _hash_dict - dictionary between hash of the index and the index in the array

    Methods
    -------
    get_hash - fetch by hash of the tuple
    get_tuple - fetch by tuple (left, hmer_number, hmer_letter, right)
    get_index - directly fetch by index in the array
    ''' 

    def __init__ (self, error_model_file: str):
        error_model = pd.read_hdf(error_model_file, key="error_model_hashed")
        hashed_idx = [ hash(x) for x in error_model.index ] 
        self.hashed_dict = dict(zip(hashed_idx, range(len(hashed_idx))))
        del hashed_idx
        self.error_model = np.array(error_model[['P(-1)', 'P(0)', 'P(+1)']])
        self.error_model = np.concatenate((self.error_model, np.zeros((1, self.error_model.shape[1]))))
        del error_model

    def hash2idx( self, hash_list):
        return [ self.hashed_dict.get(x, self.error_model.shape[0]-1) for x in hash_list]
        
    def get_hash( self, tuple_hash: int ) -> np.array : 
        hashed_idx = self.hashed_dict.get(tuple_hash, self.error_model.shape[0]-1)
        return self.error_model[hashed_idx,:]

    def get_tuple( self, tup: tuple ) -> np.array : 
        hashed_idx = self.hashed_dict.get(hash(tup), self.error_model.shape[0]-1)
        return self.error_model[hashed_idx,:]

    def get_index( self, index_list ) -> np.array : 
        return self.error_model[index_list,:]