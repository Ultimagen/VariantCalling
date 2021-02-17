import pandas as pd
import re
import numpy as np
import os
import argparse
import logging
import shutil
from os.path import join as pjoin
import glob

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


def SV_breakpoint_candidates(
    csv_filename, output_file_prefix, chunksize=10 ** 5, MQ_th=20, minappearances=5
):
    def row_function(cigar, isreverse):
        cur_id = np.array(re.findall(r"[A-Z]", cigar))
        cur_num = np.array(re.findall(r"[1-9]\d*|0", cigar)).astype(int)
        cur_align_bases = sum(cur_num[(cur_id == "M") | (cur_id == "D")])
        cur_softclip_length = (
            (cur_id[0] == "S") * cur_num[0],
            (cur_id[-1] == "S") * cur_num[-1],
        )
        if isreverse:
            cur_softclip_length = (cur_softclip_length[1], cur_softclip_length[0])
        return cur_align_bases, cur_softclip_length[0], cur_softclip_length[1]

    # # directory to put the temporary chucks in
    output_dir = f"{output_file_prefix}.temp_chuncks"
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    output_file_basename = os.path.basename(output_file_prefix)

    for ind in [f"chr{i}" for i in range(1, 24)]:  ## todo: add x,y?
        logger.info(f"Reading {csv_filename}; chunck {ind}")
        chunks = pd.read_csv(csv_filename, chunksize=chunksize, header=None, sep="\t")
        df = pd.DataFrame()
        # take only the rows of the specific chromosome
        for chunk in chunks:
            chunk.columns = [
                "read_id",
                "flag",
                "chr",
                "pos",
                "mapQ",
                "cigar",
                "SA_info",
            ]
            mask = chunk["chr"] == ind
            df = pd.concat([df, chunk[mask]])
        if df.empty:
            continue

        # df.columns = ['read_id', 'flag', 'chr', 'pos', 'mapQ', 'cigar', 'SA_info']  ## TODO

        SA_info = (
            df["SA_info"]
            .str.split(",", expand=True)
            .rename(
                columns={
                    0: "SA_chr",
                    1: "SA_chr_start",
                    2: "SA_strand",
                    3: "SA_cigar",
                    4: "SA_MQ",
                    5: "SA_unknown",
                }
            )
        )
        inds = (~SA_info.isnull()).sum(axis=1) >= 6
        df = df[inds]
        SA_info = SA_info[inds]
        df = df.join(SA_info.iloc[:, :5]).drop(columns=["SA_info"])
        df = df.astype(
            {
                "chr": "category",
                "SA_chr_start": int,
                "SA_strand": "category",
                "SA_chr": "category",
                "SA_MQ": int,
                "flag": int,
            }
        )

        df = df[(df["mapQ"] > MQ_th) & (df["SA_MQ"] > MQ_th)]
        df["isreverse"] = (
            df["flag"]
            .apply(lambda x: "{:04x}".format(x))
            .apply(lambda x: str(x)[2] == "1")
        )

        logger.info(f"cigar parsing")
        df_temp = df.apply(lambda x: row_function(x.cigar, x.isreverse), axis=1).apply(
            pd.Series
        )
        df_temp.columns = ["align_bases", "start_softclip", "end_softclip"]
        df = df.join(df_temp)

        df.reset_index(inplace=True, drop=True)
        df["chr_end"] = df["pos"] + df["align_bases"]

        df["SA_isreverse"] = df["SA_strand"] == "-"
        df["SA_chr_id"] = df["SA_chr"].apply(lambda x: x[5:])

        logger.info(f"cigar parsing SA")
        df_temp = df.apply(
            lambda x: row_function(x.SA_cigar, x.SA_isreverse), axis=1
        ).apply(pd.Series)
        df_temp.columns = ["SA_align_bases", "SA_start_softclip", "SA_end_softclip"]
        df = df.join(df_temp)

        df["argmi"] = np.argmin(
            [
                df["start_softclip"],
                df["end_softclip"],
                df["SA_start_softclip"],
                df["SA_end_softclip"],
            ],
            axis=0,
        )
        df["SA_chr_end"] = df["SA_chr_start"] + df["SA_align_bases"]

        df["firstpart"] = pd.Series(np.ones(df.shape[0]))
        # end of the first read or start of the second read
        df.loc[(df["argmi"] == 1) | (df["argmi"] == 2), "firstpart"] = 2

        df["junction_pos"] = np.repeat(None, df.shape[0])

        qri = (df["firstpart"] == 1) & (df["isreverse"] == False)
        df.loc[qri, "junction_pos"] = df["chr_end"][qri]
        qri = (df["firstpart"] == 1) & (df["isreverse"] == True)
        df.loc[qri, "junction_pos"] = df["pos"][qri]
        qri = (df["firstpart"] == 2) & (df["isreverse"] == False)
        df.loc[qri, "junction_pos"] = df["pos"][qri]
        qri = (df["firstpart"] == 2) & (df["isreverse"] == True)
        df.loc[qri, "junction_pos"] = df["chr_end"][qri]

        df["SA_junction_pos"] = np.repeat(None, df.shape[0])

        qri = (df["firstpart"] == 1) & (df["SA_isreverse"] == False)
        df.loc[qri, "SA_junction_pos"] = df["SA_chr_start"][qri]
        qri = (df["firstpart"] == 1) & (df["SA_isreverse"] == True)
        df.loc[qri, "SA_junction_pos"] = df["SA_chr_end"][qri]
        qri = (df["firstpart"] == 2) & (df["SA_isreverse"] == False)
        df.loc[qri, "SA_junction_pos"] = df["SA_chr_end"][qri]
        qri = (df["firstpart"] == 2) & (df["SA_isreverse"] == True)
        df.loc[qri, "SA_junction_pos"] = df["SA_chr_start"][qri]

        res = pd.concat(
            [
                df[
                    [
                        "read_id",
                        "chr",
                        "junction_pos",
                        "SA_chr_id",
                        "SA_junction_pos",
                        "firstpart",
                        "mapQ",
                        "SA_MQ",
                    ]
                ],
                df["chr_end"] - df["pos"],
                df["SA_chr_end"] - df["SA_chr_start"],
            ],
            axis=1,
        )

        res.columns = [
            "read_id",
            "chr",
            "junction_pos",
            "SA_chr_id",
            "SA_junction_pos",
            "firstpart",
            "mapQ",
            "SA_MQ",
            "len",
            "SA_len",
        ]

        if "-" in str(res.loc[0, "read_id"]):  # support new read_id format where sample name is pre-prepended
            res["read_id"] = res["read_id"].str.split("-", expand=True).iloc[:, 2]

        res = res.astype(
            {
                "read_id": int,
                "chr": object,
                "junction_pos": int,
                "SA_chr_id": object,
                "SA_junction_pos": int,
                "firstpart": int,
                "mapQ": int,
                "SA_MQ": int,
                "len": int,
                "SA_len": int,
            }
        )
        gb = res.groupby(
            ["chr", "junction_pos", "SA_chr_id", "SA_junction_pos"], as_index=False
        )
        final_res = gb.agg(
            {"mapQ": "mean", "SA_MQ": "mean", "len": "mean", "SA_len": "mean"}
        )

        final_res["cntf"] = gb.apply(
            lambda x: x[x["firstpart"] == 1]["firstpart"].count()
        )[None]
        final_res["cntr"] = gb.apply(
            lambda x: x[x["firstpart"] == 2]["firstpart"].count()
        )[None]

        tmpfinal_res = final_res.loc[
            final_res[["cntf", "cntr"]].min(axis=1) >= minappearances
        ]

        final_res_bed = pd.concat(
            [
                tmpfinal_res["chr"],
                tmpfinal_res["junction_pos"],
                tmpfinal_res["junction_pos"] + 10,
                tmpfinal_res["SA_junction_pos"],
            ],
            axis=1,
        )

        file_basename = pjoin(output_dir, f"{output_file_basename}_{ind}.JunctionsSV")
        logger.info(f"bed file writing {file_basename}.bed")
        if not final_res_bed.empty:
            final_res_bed.to_csv(
                f"{file_basename}.bed", sep="\t", header=False, index=False
            )

        tmpfinal_res.columns = [
            "chr_id",
            "junction_chr_pos",
            "SA_chr_id",
            "SA_junction_chr_pos",
            "mean_MQ",
            "SA_mean_MQ",
            "mean_align_bases",
            "SA_mean_align_bases",
            "F_read_cnt",
            "R_read_cnt",
        ]
        tmpfinal_res = tmpfinal_res[
            [
                "chr_id",
                "junction_chr_pos",
                "SA_chr_id",
                "SA_junction_chr_pos",
                "F_read_cnt",
                "R_read_cnt",
                "mean_MQ",
                "SA_mean_MQ",
                "mean_align_bases",
                "SA_mean_align_bases",
            ]
        ]
        logger.info(f"csv file writing {file_basename}.csv")
        if not tmpfinal_res.empty:
            tmpfinal_res.to_csv(f"{file_basename}.csv", sep="\t", index=False)

    # merging all the chunks into one
    for extension in ("csv", "bed"):
        logger.info(f"{extension} file merging")
        all_filenames = [i for i in glob.glob(f"{output_dir}/*.{extension}")]
        combined_csv = pd.concat(
            [
                pd.read_csv(
                    f, sep="\t", header=("infer" if (extension == "csv") else None)
                )
                for f in all_filenames
            ]
        )
        # export to csv
        logger.info(f"{extension} file writing {output_file_prefix}.{extension}")
        combined_csv.to_csv(
            f"{output_file_prefix}.{extension}",
            header=(extension == "csv"),
            sep="\t",
            index=False,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="sv_breakpoint_candidates.py", description="SV breakpoint candidates"
    )
    ap.add_argument(
        "--input_file", help="Name of the input csv file", type=str, required=True
    )
    ap.add_argument(
        "--output_file_prefix", help="Output file prefix", type=str, required=True
    )
    ap.add_argument("--q", "--mapq", help="Output file prefix", type=int, default=20)
    ap.add_argument(
        "--m", "--minappearances", help="Output file prefix", type=int, default=5
    )
    args = ap.parse_args()

    SV_breakpoint_candidates(
        args.input_file, args.output_file_prefix, MQ_th=args.q, minappearances=args.m,
    )
