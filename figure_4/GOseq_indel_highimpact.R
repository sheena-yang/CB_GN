################################################################################
# Fig 4 | GO enrichment of genes hit by high-impact indels
################################################################################

library(goseq)
library(dplyr)
library(ggplot2)

input_path  <- "data"
output_path <- "results"

genome           <- "hg19"
gene_length_type <- "Gene_length"

celltype_cols <- c("granule_neuron"  = "#c62f5a",
                   "cortical_neuron" = "#363636")

# genes carrying a high-impact indel
genes_gn <- c("EXOC5","ZNF628","TFEB","SLCO3A1","FAT2","DLGAP2","CAB39L","SYNPO","CEP120",
              "TWSG1","LRAT","ATG4C","PTPN2","CASR","HK3","LDB1","TK2","NLGN2","LINC00598",
              "FOXO1","NNT","CEP192","MAN2A2","MAPK6","INPP5D","SPTAN1","KRR1","PRMT3")

genes_cn <- c("IL12RB2","EXD2","UBE4B","AGBL5","TIA1","POLG","MIF-AS1","RFX3","PASK",
              "FAM161B","ZSCAN32","ALG6","DIPK1A","RALGAPA2","ST18","KCNC2","C11orf52",
              "HAVCR1","TPTE2","AXDND1","LAMB1")

################################################################################
# GOseq, correcting for gene length bias
################################################################################

gene_length <- read.delim(file.path(input_path, "hg19_refGene.length.tsv"), header = FALSE)
colnames(gene_length) <- c("gene", "transcript", "Gene_length", "Exon_length")
gene_length <- gene_length[!duplicated(gene_length$gene), ]

goseq_enrichment <- function(genes, celltype) {
  is_hit <- gene_length$gene %in% genes
  names(is_hit) <- gene_length$gene
  pwf <- nullp(is_hit, genome, bias.data = gene_length[[gene_length_type]])
  goseq(pwf, genome, "geneSymbol") %>%
    mutate(hitsPerc = numDEInCat * 100 / numInCat,
           FDR      = p.adjust(over_represented_pvalue, method = "fdr"),
           celltype = celltype)
}

go_gn <- goseq_enrichment(genes_gn, "granule_neuron")
go_cn <- goseq_enrichment(genes_cn, "cortical_neuron")

################################################################################
# Fig 4g | Enriched GO terms
################################################################################

# selected terms significantly enriched in celltypes
terms_gn <- c("carbohydrate homeostasis", "glucose homeostasis", "response to nutrient levels",
              "response to extracellular stimulus", "organelle assembly",
              "regulation of protein secretion", "regulation of protein transport")
terms_cn <- c("exonuclease activity")

go_df <- rbind(go_gn[go_gn$term %in% terms_gn, ],
               go_cn[go_cn$term %in% terms_cn, ])

go_df$neg_log10_pval <- -log10(go_df$over_represented_pvalue)
go_df <- go_df[order(go_df$neg_log10_pval), ]
go_df$term     <- factor(go_df$term, levels = go_df$term)
go_df$celltype <- factor(go_df$celltype, levels = names(celltype_cols))

p <- ggplot(go_df, aes(x = neg_log10_pval, y = term, fill = celltype)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3, width = 0.8) +
  scale_fill_manual(values = celltype_cols) +
  labs(x = expression(-log[10]("P value")), y = "", title = "enriched GO process") +
  theme_classic() +
  theme(axis.text.y      = element_text(size = 10, color = "black"),
        axis.text.x      = element_text(size = 10, color = "black"),
        axis.title.x     = element_text(size = 12, color = "black"),
        plot.title       = element_text(size = 14, color = "black", hjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line.y      = element_line(color = "black", linewidth = 0.5),
        axis.line.x      = element_line(color = "black", linewidth = 0.5))

ggsave(file.path(output_path, "fig4g_GOseq_enrichment_plot.pdf"), p,
       width = 9, height = 4, dpi = 100)
