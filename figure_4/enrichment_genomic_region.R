################################################################################
# Fig 4 | Mutation enrichment across genomic regions, observed versus permuted
################################################################################

library(data.table)
library(future.apply)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggsignif)

plan(multicore, workers = 20)

input_path  <- "data"
output_path <- "results"

n_perm <- 1000

celltype_cols <- c("granule_neuron"  = "#c62f5a",
                   "cortical_neuron" = "#363636",
                   "cortical_OL"     = "#ffbc42")

genic_categories <- c("UTR3", "UTR5", "downstream", "exonic",
                      "intronic", "splice site", "upstream")

region_levels <- c("intergenic", "genic", "upstream", "UTR5", "exonic",
                   "UTR3", "downstream", "splice site", "intronic")

donor_gn       <- function(cell) substr(cell, 1, 4)
donor_cortical <- function(cell) sub("[-_].*", "", cell)

classify_func <- function(func) {
  fcase(
    func %in% "intergenic",                     "intergenic",
    func %in% "upstream",                       "upstream",
    func %in% "downstream",                     "downstream",
    func %in% "UTR5",                           "UTR5",
    func %in% "UTR3",                           "UTR3",
    func %in% c("intronic", "ncRNA_intronic"),  "intronic",
    func %in% c("exonic", "ncRNA_exonic"),      "exonic",
    func %in% c("splicing", "ncRNA_splicing"),  "splice site",
    default = "others"
  )
}

################################################################################
# Observed and permuted mutations per region
################################################################################

count_observed <- function(annovar_path, muttype, donor_fun) {
  files <- list.files(annovar_path, pattern = paste0("pass_", muttype, ".hg19_multianno.csv$"),
                      full.names = TRUE, recursive = TRUE)
  dt_list <- lapply(files, function(f) {
    dt   <- fread(f, select = "Func.refGene")
    cell <- basename(dirname(f))
    dt[, func_cat := classify_func(Func.refGene)]
    dt[, .N, by = func_cat][, `:=`(cell = cell, donor = donor_fun(cell))]
  })
  dcast(rbindlist(dt_list), cell + donor ~ func_cat, value.var = "N", fill = 0)
}

count_permuted <- function(perm_path, muttype, donor_fun) {
  files <- list.files(perm_path, pattern = paste0("perms_", muttype, ".hg19_multianno.csv$"),
                      full.names = TRUE, recursive = TRUE)
  dt_list <- future_lapply(files, function(f) {
    dt <- fread(f, select = "Func.refGene")
    stopifnot(nrow(dt) %% n_perm == 0)
    dt[, perm.id := rep(seq_len(n_perm), each = nrow(dt) / n_perm)]
    cell <- basename(dirname(f))
    dt[, func_cat := classify_func(Func.refGene)]
    dt[, .N, by = .(perm.id, func_cat)][, `:=`(cell = cell, donor = donor_fun(cell))]
  })
  dcast(rbindlist(dt_list), cell + donor + perm.id ~ func_cat, value.var = "N", fill = 0)
}

# regions absent from a cell type are counted as zero
fill_missing_cols <- function(dt, all_cols) {
  for (col in setdiff(all_cols, colnames(dt))) dt[, (col) := 0]
  dt[, ..all_cols]
}

enrichment_by_donor <- function(observed, permuted) {
  region_cols <- setdiff(colnames(observed), c("cell", "donor", "others"))
  num_obs  <- observed[, lapply(.SD, sum), by = donor, .SDcols = region_cols]
  num_perm <- permuted[, lapply(.SD, sum), by = donor, .SDcols = region_cols]
  ratio <- merge(num_perm, num_obs, by = "donor", suffixes = c("_perm", "_obs"))
  for (col in region_cols) {
    ratio[[col]] <- ratio[[paste0(col, "_obs")]] * n_perm / ratio[[paste0(col, "_perm")]]
  }
  ratio
}

region_enrichment <- function(muttype) {

  observed <- list(
    granule_neuron  = count_observed(file.path(input_path, "annovar/granule_neuron"), muttype, donor_gn),
    cortical_neuron = count_observed(file.path(input_path, "annovar/cortical_neuron"), muttype, donor_cortical),
    cortical_OL     = count_observed(file.path(input_path, "annovar/cortical_OL"), muttype, donor_cortical)
  )
  permuted <- list(
    granule_neuron  = count_permuted(file.path(input_path, "perm/granule_neuron"), muttype, donor_gn),
    cortical_neuron = count_permuted(file.path(input_path, "perm/cortical_neuron"), muttype, donor_cortical),
    cortical_OL     = count_permuted(file.path(input_path, "perm/cortical_OL"), muttype, donor_cortical)
  )

  all_cols <- unique(unlist(lapply(observed, colnames)))
  observed <- lapply(observed, fill_missing_cols, all_cols = all_cols)

  for (dt in c(observed, permuted)) {
    dt[, genic := rowSums(.SD), .SDcols = genic_categories]
  }

  # regions with at least 20 mutations in one of the cell types
  region_cats  <- setdiff(all_cols, c("cell", "donor", "others"))
  keep_regions <- region_cats[sapply(region_cats, function(region) {
    max(sapply(observed, function(dt) sum(dt[[region]], na.rm = TRUE))) >= 20
  })]
  keep_regions <- region_levels[region_levels %in% c(keep_regions, "genic")]

  ratio <- Map(enrichment_by_donor, observed, permuted)
  for (ct in names(ratio)) ratio[[ct]]$celltype <- ct

  bind_rows(ratio) %>%
    dplyr::select(celltype, all_of(keep_regions)) %>%
    pivot_longer(cols = -celltype, names_to = "region", values_to = "enrichment") %>%
    mutate(region = factor(region, levels = keep_regions))
}

plot_enrichment <- function(df_long, y_label, y_max) {
  ggplot(df_long, aes(x = celltype, y = enrichment, color = celltype)) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    geom_hline(yintercept = 1, color = "black", linewidth = 0.4) +
    geom_signif(comparisons = list(c("granule_neuron", "cortical_neuron"),
                                   c("granule_neuron", "cortical_OL")),
                map_signif_level = TRUE, step_increase = 0.12, test = "wilcox.test") +
    facet_wrap(~ region, nrow = 1, scales = "free_x") +
    scale_color_manual(values = celltype_cols) +
    coord_cartesian(ylim = c(0, y_max)) +
    labs(x = "Genome region", y = y_label, color = "Cell type") +
    theme_classic(base_size = 14) +
    theme(strip.background = element_blank(),
          strip.text       = element_text(size = 12),
          axis.text.x      = element_text(angle = 45, hjust = 1),
          legend.position  = "right")
}

################################################################################
# Fig 4a | SNV enrichment by genomic region
################################################################################

snv_long <- region_enrichment("snv")

ggsave(file.path(output_path, "fig4ab_snv_enrichment_genomic_region.pdf"),
       plot = plot_enrichment(snv_long, "sSNV enrichment (obs/exp)", 2),
       width = 10, height = 4)

################################################################################
# Fig 4b | Indel enrichment by genomic region
################################################################################

indel_long <- region_enrichment("indel")

ggsave(file.path(output_path, "fig4cd_indel_enrichment_genomic_region.pdf"),
       plot = plot_enrichment(indel_long, "sIndel enrichment (obs/exp)", 3),
       width = 6.5, height = 4)
