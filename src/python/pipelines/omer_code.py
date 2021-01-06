
import pandas as pd
import re
import numpy as np
import os

def SV_junction_detection(csv_filename):
    #samtools view /data/201220_structure_variants_omer_code/150300-BC03.aligned.sorted.duplicates_marked.bam chr8 | grep SA | awk '$5>20 {print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$12}' > 150300-BC03.aligned.sorted.duplicates_marked.bam.SA.csv &
    #cmd = ['samtools','view',bam_file,'|','grep','SA','|','awk','\'$5>20 {print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$12}\'','>', csv_filename]
    #subprocess.check_call(cmd)
df = pd.read_csv(csv_filename, sep='\t')
df.columns = ['read_id', 'flag', 'chr', 'pos', 'mapQ', 'cigar', 'SA_info']  ## TODO
cnt = 0


# filter df
SA_info = df['SA_info'].str.split(',', expand = True).rename(columns={0:'SA_chr',1:'SA_chr_start',2:'SA_strand',3:'SA_cigar',4:'SA_MQ',5:'SA_unknown'})
inds = (~SA_info.isnull()).sum(axis=1) >= 6  ## todo: ASk Omer. -  6 or >=6
df = df[inds]
SA_info = SA_info[inds]
df = df.join(SA_info.iloc[:, :5]).drop(columns=['SA_info'])
df = df.astype({
    'chr': 'category',
    'SA_chr_start': int,
    'SA_strand': 'category',
    'SA_chr': 'category',
    'SA_MQ': int
})

MQ_th = 20
minappearance = 5
df = df[(df['mapQ'] > MQ_th) & (df['SA_MQ'] > MQ_th)]
df['isreverse'] = df['flag'].apply(lambda x: "{:04x}".format(x)).apply(lambda x: str(x)[2] == '1')

def row_function(cigar, isreverse):
    cur_id = np.array(re.findall(r'[A-Z]', cigar))
    cur_num = np.array(re.findall(r'[1-9]\d*|0', cigar)).astype(int)
    cur_align_bases = (sum(cur_num[(cur_id == 'M') | (cur_id == 'D')]))
    cur_softclip_length = ((cur_id[0] == 'S') * cur_num[0], (cur_id[-1] == 'S') * cur_num[-1])
    if isreverse:
        cur_softclip_length = (cur_softclip_length[1], cur_softclip_length[0])
    return cur_align_bases, cur_softclip_length[0], cur_softclip_length[1]

df_temp = df.apply(lambda x: row_function(x.cigar, x.isreverse), axis=1).apply(pd.Series)
df_temp.columns = ['align_bases', 'start_softclip', 'end_softclip']
df = df.join(df_temp)

df.reset_index(inplace=True,drop=True)
df['chr_end'] = df['pos'] + df['align_bases']

df['SA_isreverse'] = df['SA_strand'] == '-'
df['SA_chr_id'] = df['SA_chr'].apply(lambda x: x[5:])

df_temp = df.apply(lambda x: row_function(x.SA_cigar, x.SA_isreverse), axis=1).apply(pd.Series)
df_temp.columns = ['SA_align_bases', 'SA_start_softclip', 'SA_end_softclip']
df = df.join(df_temp)

df['argmi'] = (np.argmin([df['start_softclip'],df['end_softclip'], df['SA_start_softclip'],df['SA_end_softclip']], axis = 0))
df['SA_chr_end'] = df['SA_chr_start'] + df['SA_align_bases']

df['firstpart'] = pd.Series(np.ones(df.shape[0]))
    # end of the first read or start of the second read
df.loc[(df['argmi']  == 1) | (df['argmi']  == 2),'firstpart'] = 2 ### TODO: check numbers?? change to bool?

df['junction_pos'] = np.repeat(None, df.shape[0])
# funciton??
qri = (df['firstpart'] == 1) & (df['isreverse'] == False);df.loc[qri,'junction_pos'] = df['chr_end'][qri]
qri = (df['firstpart'] == 1) & (df['isreverse'] == True);df.loc[qri,'junction_pos'] = df['pos'][qri]
qri = (df['firstpart'] == 2) & (df['isreverse'] == False);df.loc[qri,'junction_pos'] = df['pos'][qri]
qri = (df['firstpart'] == 2) & (df['isreverse'] == True);df.loc[qri,'junction_pos'] = df['chr_end'][qri]

df['SA_junction_pos'] = np.repeat(None, df.shape[0])

qri = (df['firstpart'] == 1) & (df['SA_isreverse'] == False);df.loc[qri,'SA_junction_pos'] = df['SA_chr_start'][qri]
qri = (df['firstpart'] == 1) & (df['SA_isreverse'] == True);df.loc[qri,'SA_junction_pos'] = df['SA_chr_end'][qri]
qri = (df['firstpart'] == 2) & (df['SA_isreverse'] == False);df.loc[qri,'SA_junction_pos'] = df['SA_chr_end'][qri]
qri = (df['firstpart'] == 2) & (df['SA_isreverse'] == True);df.loc[qri,'SA_junction_pos'] = df['SA_chr_start'][qri]

res = pd.concat([df[['read_id','chr','junction_pos','SA_chr_id','SA_junction_pos','firstpart','mapQ','SA_MQ']],
                   df['chr_end'] - df['pos'],
                   df['SA_chr_end'] - df['SA_chr_start']],axis=1)
# res = pd.DataFrame([df['read_id'], df['chr'], df['junction_pos'], df['SA_chr_id'], df['SA_junction_pos'], df['firstpart'], df['mapQ'], df['SA_MQ'],
#                     df['chr_end'] - df['pos'],
#                     df['SA_chr_end'] - df['SA_chr_start']]).transpose()
res.columns = ['read_id', 'chr', 'junction_pos', 'SA_chr_id', 'SA_junction_pos', 'firstpart', 'mapQ', 'SA_MQ',
               'len', 'SA_len']

res = res.astype({ ## TODO: remove this one
    'read_id':int,
    'chr':object,
    'junction_pos':int,
    'SA_chr_id':object,
    'SA_junction_pos':int,
    'firstpart': int,
    'mapQ': int,
    'SA_MQ': int,
    'len': int,
    'SA_len': int
})
gb = res.groupby(['chr', 'junction_pos', 'SA_chr_id','SA_junction_pos'], as_index = False)
final_res = gb.agg({'mapQ': 'mean', 'SA_MQ':'mean', 'len':'mean', 'SA_len':'mean'}) ##TODO: returns df

final_res['cntf'] = gb.apply(lambda x: x[x['firstpart'] == 1]['firstpart'].sum())[None]
final_res['cntr'] = gb.apply(lambda x: x[x['firstpart'] == 2]['firstpart'].sum())[None] # TODO: if book sum - count

basename_file = os.path.splitext(csv_filename)

tmpfinal_res = final_res.loc[final_res[['cntf','cntr']].min(axis=1) > minappearance]

final_res_bed = pd.concat([tmpfinal_res['chr'],tmpfinal_res['junction_pos'], tmpfinal_res['junction_pos']+10,tmpfinal_res['SA_junction_pos']], axis=1)

final_res_bed.to_csv(basename_file[0]+".JunctionsSV.bed", sep='\t', header = False, index=False)

tmpfinal_res.columns = ['chr_id','junction_chr_pos','SA_chr_id','SA_junction_chr_pos','mean_MQ','SA_mean_MQ','mean_align_bases','SA_mean_align_bases','F_read_cnt','R_read_cnt']
tmpfinal_res = tmpfinal_res[['chr_id','junction_chr_pos','SA_chr_id','SA_junction_chr_pos','F_read_cnt','R_read_cnt','mean_MQ','SA_mean_MQ','mean_align_bases','SA_mean_align_bases']]
tmpfinal_res.to_csv(basename_file[0]+".JunctionsSV.csv", sep='\t', index=False)


csv_filename = '/data/201220_structure_variants_omer_code/150300-BC03.aligned.sorted.duplicates_marked.bam.SA.csv'
SV_junction_detection(csv_filename)


##debugging
mine = pd.read_csv(basename_file[0]+".JunctionsSV.csv")
omer = pd.read_csv(basename_file[0]+".JunctionsSV.csv")