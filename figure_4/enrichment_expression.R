################################################################################
# Fig 4 | sSNV and sIndel enrichment in relation to gene expression
################################################################################

library(Seurat)
library(data.table)
library(future.apply)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggpubr)

plan(multicore, workers = 20)

input_path  <- "data"
output_path <- "results"

n_bins  <- 5
n_cells <- 5000
n_perm  <- 1000

decile_cols <- paste0("decile", 1:n_bins)

celltype_cols <- c("granule_neuron"  = "#c62f5a",
                   "cortical_neuron" = "#363636",
                   "cortical_OL"     = "#ffbc42")

# mutations assigned to a gene
genic_func <- c("exonic", "exonic;splicing", "intronic", "splicing", "UTR3", "UTR5", "UTR5;UTR3")

donor_gn       <- function(cell) substr(cell, 1, 4)
donor_cortical <- function(cell) sub("[-_].*", "", cell)

################################################################################
# Gene expression bins from snRNA-seq
################################################################################

set.seed(123)

expression_bins <- function(average_expr, genes) {
  data.frame(average_expr = as.numeric(average_expr), Gene.refGene = genes) %>%
    mutate(decile = as.factor(ntile(average_expr, n = n_bins)))
}

subsample <- function(sce) subset(sce, cells = sample(colnames(sce), n_cells))

sce_gn <- readRDS(file.path(input_path, "sce_GN.RDS"))
sce_gn <- subsample(subset(sce_gn, subset = cluster == "Granule"))
avg_gn <- AverageExpression(sce_gn, group.by = "cluster", slot = "data")$RNA
expr_gn <- expression_bins(avg_gn, rownames(avg_gn))

# published prefrontal cortex snRNA-seq
sce_ref <- readRDS(file.path(input_path, "Jeffries_Nature2025_PFC_snRNA.rds"))

expression_bins_ref <- function(label) {
  sce <- subsample(subset(sce_ref, subset = new_clusters3 == label))
  expression_bins(Matrix::rowMeans(sce@assays$RNA@layers$data), rownames(sce))
}

expr_cn <- expression_bins_ref("Excitotary_Neurons")
expr_ol <- expression_bins_ref("Oligodendrocytes")

################################################################################
# Observed and permuted mutations per expression bin
################################################################################

bin_counts <- function(dt, expr_df) {
  dt %>%
    filter(Func.refGene %in% genic_func) %>%
    mutate(Gene.refGene = sub(";.*$", "", Gene.refGene)) %>%
    filter(Gene.refGene != "") %>%
    left_join(expr_df, by = "Gene.refGene")
}

count_observed <- function(annovar_path, muttype, expr_df, donor_fun) {
  files <- list.files(annovar_path, pattern = paste0("pass_", muttype, ".hg19_multianno.csv$"),
                      full.names = TRUE, recursive = TRUE)
  dt_list <- lapply(files, function(f) {
    cell <- basename(dirname(f))
    fread(f, select = c("Func.refGene", "Gene.refGene")) %>%
      bin_counts(expr_df) %>%
      count(decile) %>%
      pivot_wider(names_from = decile, values_from = n, names_prefix = "decile", values_fill = 0) %>%
      mutate(cell = cell, donor = donor_fun(cell))
  })
  bind_rows(dt_list) %>%
    select(cell, donor, all_of(decile_cols)) %>%
    mutate(across(starts_with("decile"), ~replace_na(., 0)))
}

count_permuted <- function(perm_path, muttype, expr_df, donor_fun) {
  files <- list.files(perm_path, pattern = paste0("perms_", muttype, ".hg19_multianno.csv$"),
                      full.names = TRUE, recursive = TRUE)
  dt_list <- future_lapply(files, function(f) {
    dt <- fread(f, select = c("Func.refGene", "Gene.refGene"))
    stopifnot(nrow(dt) %% n_perm == 0)
    dt[, perm.id := rep(seq_len(n_perm), each = nrow(dt) / n_perm)]
    cell <- basename(dirname(f))
    dt %>%
      bin_counts(expr_df) %>%
      count(perm.id, decile) %>%
      pivot_wider(names_from = decile, values_from = n, names_prefix = "decile", values_fill = 0) %>%
      mutate(cell = cell, donor = donor_fun(cell))
  })
  bind_rows(dt_list) %>%
    select(cell, donor, perm.id, all_of(decile_cols)) %>%
    mutate(across(starts_with("decile"), ~replace_na(., 0)))
}

expression_enrichment <- function(muttype) {

  observed <- list(
    granule_neuron  = count_observed(file.path(input_path, "annovar/granule_neuron"), muttype, expr_gn, donor_gn),
    cortical_neuron = count_observed(file.path(input_path, "annovar/cortical_neuron"), muttype, expr_cn, donor_cortical),
    cortical_OL     = count_observed(file.path(input_path, "annovar/cortical_OL"), muttype, expr_ol, donor_cortical)
  )
  permuted <- list(
    granule_neuron  = count_permuted(file.path(input_path, "perm/granule_neuron"), muttype, expr_gn, donor_gn),
    cortical_neuron = count_permuted(file.path(input_path, "perm/cortical_neuron"), muttype, expr_cn, donor_cortical),
    cortical_OL     = count_permuted(file.path(input_path, "perm/cortical_OL"), muttype, expr_ol, donor_cortical)
  )

  ratio <- Map(function(obs, perm, celltype) {
    obs  <- obs %>% summarise(across(all_of(decile_cols), ~sum(.x, na.rm = TRUE) * n_perm))
    perm <- perm %>% summarise(across(all_of(decile_cols), ~sum(.x, na.rm = TRUE)))
    obs[decile_cols] <- obs[decile_cols] / perm[decile_cols]
    obs %>% mutate(celltype = celltype)
  }, observed, permuted, names(observed))

  bind_rows(ratio) %>%
    pivot_longer(cols = starts_with("decile"), names_to = "decile", values_to = "ratio") %>%
    mutate(decile_num = as.integer(gsub("decile", "", decile)),
           celltype   = factor(celltype, levels = names(celltype_cols)))
}

plot_enrichment <- function(df_long, y_label) {
  ggplot(df_long, aes(x = decile_num, y = ratio, color = celltype, fill = celltype)) +
    stat_cor(aes(group = celltype), method = "pearson", size = 4,
             show.legend = FALSE, label.x.npc = "right", hjust = 1) +
    geom_smooth(method = "lm", se = TRUE, linewidth = 1.3, alpha = 0.08) +
    geom_point(size = 2) +
    scale_color_manual(values = celltype_cols) +
    scale_fill_manual(values = celltype_cols) +
    scale_x_continuous(breaks = 1:n_bins, labels = 1:n_bins) +
    labs(x = "Gene expression level", y = y_label) +
    theme_classic(base_size = 14) +
    theme(legend.title = element_blank(),
          axis.text.x  = element_text(size = 12),
          axis.text.y  = element_text(size = 12))
}

################################################################################
# Fig 4h | sSNV enrichment in relation to gene expression
################################################################################

ggsave(file.path(output_path, "fig4h_snv_enrichment_expression.pdf"),
       plot = plot_enrichment(expression_enrichment("snv"), "sSNV enrichment (obs/exp)"),
       width = 5, height = 4)

################################################################################
# Fig 4h | sIndel enrichment in relation to gene expression
################################################################################

ggsave(file.path(output_path, "fig4h_indel_enrichment_expression.pdf"),
       plot = plot_enrichment(expression_enrichment("indel"), "sIndel enrichment (obs/exp)"),
       width = 5, height = 4)
