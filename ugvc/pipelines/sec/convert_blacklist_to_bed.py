#!/env/python
import pandas as pd

# Copyright 2022 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Convert blacklist in h5 file to BED
# CHANGELOG in reverse chronological order

if __name__ == "__main__":
    bl = pd.read_hdf("blacklist_ua_good_old_blacklist.h5")
    bl = bl.reset_index()["index"]
    chrom = bl.apply(lambda x: x[0])
    pos = bl.apply(lambda x: x[1])
    df = pd.DataFrame({"chrom": chrom, "start": pos - 1, "end": pos})
    df.to_csv("blacklist.bed", sep="\t", index=False)
