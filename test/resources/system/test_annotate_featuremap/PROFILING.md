Run the test to create the files:

    pytest -sv test/system/test_annotate_featuremap.py

Edit the file and split into header and body:

    cp tp_featuremap_chr20.vcf tp_featuremap_chr20.vcf.header
    cp tp_featuremap_chr20.vcf tp_featuremap_chr20.vcf.body

Concatenate the body a bunch of times (yes, should be a script):

    cat tp_featuremap_chr20.vcf.body tp_featuremap_chr20.vcf.body > tp_featuremap_chr20.vcf.body2
    cat tp_featuremap_chr20.vcf.body2 tp_featuremap_chr20.vcf.body2> tp_featuremap_chr20.vcf.body4
    cat tp_featuremap_chr20.vcf.body4 tp_featuremap_chr20.vcf.body4> tp_featuremap_chr20.vcf.body8
    cat tp_featuremap_chr20.vcf.body8 tp_featuremap_chr20.vcf.body8> tp_featuremap_chr20.vcf.body16
    cat tp_featuremap_chr20.vcf.body16 tp_featuremap_chr20.vcf.body16> tp_featuremap_chr20.vcf.body32
    cat tp_featuremap_chr20.vcf.body32 tp_featuremap_chr20.vcf.body32> tp_featuremap_chr20.vcf.body64
    cat tp_featuremap_chr20.vcf.body64 tp_featuremap_chr20.vcf.body64> tp_featuremap_chr20.vcf.body128
    cat tp_featuremap_chr20.vcf.body128 tp_featuremap_chr20.vcf.body128> tp_featuremap_chr20.vcf.body256
    cat tp_featuremap_chr20.vcf.body256 tp_featuremap_chr20.vcf.body256> tp_featuremap_chr20.vcf.body512
    cat tp_featuremap_chr20.vcf.body512 tp_featuremap_chr20.vcf.body512> tp_featuremap_chr20.vcf.body1024
    cat tp_featuremap_chr20.vcf.body1024 tp_featuremap_chr20.vcf.body1024> tp_featuremap_chr20.vcf.body2048
    cat tp_featuremap_chr20.vcf.body2048 tp_featuremap_chr20.vcf.body2048> tp_featuremap_chr20.vcf.body4096

Make sure records are sorted:

    sort tp_featuremap_chr20.vcf.body4096 > tp_featuremap_chr20.vcf.body4096.sorted

Add header:

    cat tp_featuremap_chr20.vcf.header tp_featuremap_chr20.vcf.body4096.sorted >tp_featuremap_chr20_dup.vcf

Compress and index:

    bgzip tp_featuremap_chr20_dup.vcf
    tabix tp_featuremap_chr20_dup.vcf.gz

The above files can be found here:



Profile!

    sudo py-spy record -- python ugvc/pipelines/vcfbed/annotate_contig.py --vcf_in /Users/gavrie/source/ultimagen/VariantCalling/test/resources/system/test_annotate_featuremap/tp_featuremap_chr20_dup.vcf.gz --vcf_out /private/tmp/pytest-of-gavrie/pytest-1/test_annotate_featuremap0/chr20.vcf.gz --annotators_pickle /private/tmp/pytest-of-gavrie/pytest-1/test_annotate_featuremap0/tp_featuremap_chr20_dup.annotated.vcf.gz.annotators.pickle --contig chr20 --chunk_size 10000