# CB_GN

Analysis code for somatic mutation analysis of human cerebellar granule neurons.

## Repository Structure

```
CB_GN/
├── README.md
│
├── figure_1/                    # Cell type and burden analysis
│   ├── scRNA.R                  # snRNA-seq analysis
│   ├── snv_burden.R             # sSNV burden analysis
│   └── indel_burden.R           # sIndel burden analysis
│
├── figure_2/                    # sSNV spectrum analysis
│   ├── snv_PCA.R                # sSNV PCA
│   └── snv_signature.R          # sSNV signature analysis
│
├── figure_3/                    # sIndel spectrum analysis
│   ├── indel_PCA.R              # sIndel PCA
│   └── indel_signature.R        # sIndel signature analysis
│
├── figure_4/                    # Enrichment and annotation analysis
│   ├── run_GN_ANNOVAR.sh        # ANNOVAR annotation
│   ├── run_GN_snpeff.sh         # snpEff annotation
│   ├── enrichment_genomic_region.R  # Genomic region analysis
│   ├── snpeff_impact.R          # Mutation impact analysis
│   ├── GOseq_indel_highimpact.R # GO enrichment analysis
│   ├── enrichment_expression.R  # Expression enrichment analysis
│   ├── enrichment_replication_timing.R # Replication timing enrichment analysis
│   └── enrichment_accessibility.R # Accessibility enrichment analysis
│
├── figure_5/                    # Disease and ID4 analysis
│   ├── burden_comparison.R      # Burden comparison
│   ├── RF_model_ID4_group_prediction.R # ID4 prediction
│   └── cerebellum_GN_ID4_heatmap.R # ID4 heatmap
│
└── figure_6/                    # Lineage reconstruction analysis
    ├── infinite_sites_denoise.py # Variant-by-cell genotype matrix denoising
    ├── plot_lineage_tree_snv_burden.py # Lineage tree plot
    ├── plot_ultrametric_tree.py # Ultrametric tree plot
    ├── plot_mrca_lollipop.py    # MRCA plot
    ├── region_mixing_deviation_index.py # Region mixing analysis
    ├── LTT_analysis.py          # LTT analysis
    └── *.newick                 # Tree files
```

## Requirements

Operating system: Linux; tested on Rocky Linux 9.8 (Blue Onyx), x86_64

R: v4.4.3, Seurat (v4.3.0), scDblFinder (v1.16.0), Harmony (v1.2.3),
scan2 (v1.0), MutationalPatterns (v3.14.0), pracma (v2.4.4), GOseq (v1.54.0),
lme4 (v1.1.36), lmerTest (v3.1.3), emmeans (v2.0.0), MosaicHunter (v0.1.4),
and rtreefit (v1.2.0)

Python: v3.12.7, NumPy (v1.26.4), pandas (v2.2.2), and Matplotlib (v3.9.2)
