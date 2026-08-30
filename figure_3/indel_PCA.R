################################################################################
# Fig 3 | PCA of single-cell indel spectra
################################################################################

library(MutationalPatterns)
library(BSgenome)
library(dplyr)
library(ggplot2)

ref_genome <- "BSgenome.Hsapiens.UCSC.hg19"
library(ref_genome, character.only = TRUE)

input_path  <- "data"
output_path <- "results"

load_indel_matrix <- function(vcf_dir, pattern) {
  vcf_files    <- list.files(path = vcf_dir, pattern = pattern, full.names = TRUE)
  sample_names <- unlist(strsplit(basename(vcf_files), "\\."))[3 * seq_along(vcf_files) - 1]
  grl <- read_vcfs_as_granges(vcf_files, sample_names, ref_genome, type = "indel")
  seqlengths(grl) <- seqlengths(Hsapiens)[1:22][seqlevels(grl)]
  count_indel_contexts(get_indel_context(grl, ref_genome))
}

# granule neuron
mut_mat_gn_indel <- load_indel_matrix(file.path(input_path, "vcf/granule_neuron"), "^sindel_list")

# published cortical neurons and OLs
mut_mat_neuron_indel <- load_indel_matrix(file.path(input_path, "vcf/cortical_neuron"), "cortical_neuron_indel")
mut_mat_ol_indel     <- load_indel_matrix(file.path(input_path, "vcf/cortical_OL"), "cortical_OL_indel")

################################################################################
# Cell metadata
################################################################################

df_burden       <- read.csv(file.path(input_path, "indel_burden.csv"))
burden_cortical <- read.csv(file.path(input_path, "Ganz_Cell2024_indel_burden.csv"))
burden_cortical <- burden_cortical[burden_cortical$outlier == "NORMAL", ]
burden_m        <- rbind(df_burden, burden_cortical)

mut_mat_gn_indel     <- mut_mat_gn_indel[, intersect(colnames(mut_mat_gn_indel), burden_m$cell_id)]
mut_mat_neuron_indel <- mut_mat_neuron_indel[, intersect(colnames(mut_mat_neuron_indel), burden_m$cell_id)]
mut_mat_ol_indel     <- mut_mat_ol_indel[, intersect(colnames(mut_mat_ol_indel), burden_m$cell_id)]

# keep cells with at least 10 indels
mut_mat_gn_indel     <- mut_mat_gn_indel[, colSums(mut_mat_gn_indel, na.rm = TRUE) >= 10]
mut_mat_neuron_indel <- mut_mat_neuron_indel[, colSums(mut_mat_neuron_indel, na.rm = TRUE) >= 10]
mut_mat_ol_indel     <- mut_mat_ol_indel[, colSums(mut_mat_ol_indel, na.rm = TRUE) >= 10]

################################################################################
# PCA
################################################################################

mut_mat_all <- as.matrix(cbind(mut_mat_gn_indel, mut_mat_neuron_indel, mut_mat_ol_indel))
storage.mode(mut_mat_all) <- "numeric"

# drop indel contexts that are constant across cells
keep_rows <- apply(mut_mat_all, 1, function(x) {
  x <- x[is.finite(x)]
  length(x) > 1 && sd(x) > 0
})
mut_mat_all <- mut_mat_all[keep_rows, , drop = FALSE]

pca <- prcomp(t(mut_mat_all), center = TRUE, scale. = TRUE)

pca_df <- as.data.frame(pca$x[, 1:2, drop = FALSE])
pca_df$cell_id <- rownames(pca_df)

plot_df <- pca_df %>%
  left_join(burden_m %>% select(cell_id, age, celltype), by = "cell_id") %>%
  mutate(age_group = case_when(age <= 2              ~ "Infant",
                               age >= 15 & age <= 20 ~ "Adolescent",
                               age >= 30 & age <= 70 ~ "Adult",
                               age > 70              ~ "Aged"),
         age_group = factor(age_group, levels = c("Infant", "Adolescent", "Adult", "Aged")))

var_exp <- summary(pca)$importance[2, 1:2]

################################################################################
# Fig 3b
################################################################################

celltype_cols <- c("granule_neuron"  = "#c62f5a",
                   "cortical_neuron" = "#363636",
                   "cortical_OL"     = "#ffbc42")

shape_vec <- c(Infant = 21, Adolescent = 22, Adult = 24, Aged = 23)

p <- ggplot(plot_df, aes(x = PC1, y = PC2, fill = celltype, shape = age_group)) +
  geom_point(size = 3.5, colour = "grey20", stroke = 0.4) +
  scale_fill_manual(values = celltype_cols, name = "Cell type") +
  scale_shape_manual(values = shape_vec, name = "Age group", na.translate = FALSE) +
  guides(fill  = guide_legend(order = 1,
                              override.aes = list(shape = 21, size = 4,
                                                  colour = "grey20", stroke = 0.4)),
         shape = guide_legend(order = 2,
                              override.aes = list(fill = "white", colour = "grey20",
                                                  size = 4, stroke = 0.4))) +
  labs(x = paste0("PC1 (", round(var_exp[1] * 100, 1), "%)"),
       y = paste0("PC2 (", round(var_exp[2] * 100, 1), "%)")) +
  theme_classic(base_size = 14) +
  theme(axis.title   = element_text(size = 14),
        axis.text    = element_text(size = 12, colour = "black"),
        legend.title = element_text(size = 12),
        legend.text  = element_text(size = 11))

ggsave(file.path(output_path, "fig3b_indel_pca.pdf"), plot = p, width = 6.2, height = 5)
