################################################################################
# Fig 4 | sSNV and sIndel enrichment in relation to chromatin accessibility
################################################################################

library(data.table)
library(future.apply)
library(GenomicRanges)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggpubr)

plan(multicore, workers = 20)

input_path  <- "data"
output_path <- "results"

n_bins <- 5
n_perm <- 1000

autosomes <- paste0("chr", 1:22)

celltype_cols <- c("granule_neuron"  = "#c62f5a",
                   "cortical_neuron" = "#363636",
                   "cortical_OL"     = "#ffbc42")

################################################################################
# Accessibility bins from snATAC-seq
################################################################################

accessibility_bins <- function(atac_file) {
  atac <- fread(file.path(input_path, atac_file), header = FALSE,
                col.names = c("name", "size", "covered", "sum", "mean0", "mean"))
  atac <- separate(atac, name, into = c("chr", "start", "end"), sep = "_", convert = TRUE) %>%
    filter(chr %in% autosomes) %>%
    mutate(start = suppressWarnings(as.integer(start)),
           end   = suppressWarnings(as.integer(end))) %>%
    filter(!is.na(start), !is.na(end)) %>%
    mutate(bin = ifelse(mean0 == 0, 1L, NA_integer_))
  open_bins <- which(atac$mean0 != 0)
  atac$bin[open_bins] <- ntile(atac$mean0[open_bins], n_bins - 1) + 1
  atac
}

atac_gn <- accessibility_bins("atac/granule_neuron.hg19.1kb.txt")
atac_cn <- accessibility_bins("atac/cortical_neuron.hg19.1kb.txt")
atac_ol <- accessibility_bins("atac/cortical_OL.hg19.1kb.txt")

################################################################################
# Observed and permuted mutations per accessibility bin
################################################################################

assign_bin <- function(mut_df, atac_df) {
  atac_gr <- GRanges(seqnames = atac_df$chr,
                     ranges = IRanges(start = atac_df$start + 1, end = atac_df$end),
                     bin = atac_df$bin)
  mut_gr <- GRanges(seqnames = paste0("chr", mut_df$chr),
                    ranges = IRanges(start = mut_df$start, end = mut_df$end))
  hits <- findOverlaps(mut_gr, atac_gr)
  mut_df$bin <- NA_integer_
  mut_df$bin[queryHits(hits)] <- mcols(atac_gr)$bin[subjectHits(hits)]
  mut_df
}

read_observed <- function(annovar_path, muttype) {
  files <- list.files(file.path(input_path, annovar_path),
                      pattern = paste0("pass_", muttype, ".hg19_multianno.txt$"),
                      full.names = TRUE, recursive = TRUE)
  bind_rows(lapply(files, function(f) {
    fread(f, select = c("Chr", "Start", "End"),
          col.names = c("chr", "start", "end"))
  }))
}

read_permuted <- function(perm_path, muttype) {
  files <- list.files(file.path(input_path, perm_path),
                      pattern = paste0("perms_", muttype, ".hg19_multianno.txt$"),
                      full.names = TRUE, recursive = TRUE)
  bind_rows(future_lapply(files, function(f) {
    dt <- fread(f, select = c("Chr", "Start", "End"),
                col.names = c("chr", "start", "end"))
    stopifnot(nrow(dt) %% n_perm == 0)
    dt[, perm.id := rep(seq_len(n_perm), each = nrow(dt) / n_perm)]
    dt
  }))
}

accessibility_enrichment <- function(muttype) {

  atac <- list(granule_neuron = atac_gn, cortical_neuron = atac_cn, cortical_OL = atac_ol)

  ratio <- lapply(names(atac), function(celltype) {

    observed <- assign_bin(read_observed(file.path("annovar", celltype), muttype), atac[[celltype]])
    permuted <- assign_bin(read_permuted(file.path("perm", celltype), muttype), atac[[celltype]])

    obs <- observed %>% filter(!is.na(bin)) %>% count(bin, name = "obs")
    exp <- permuted %>% filter(!is.na(bin)) %>%
      count(perm.id, bin, name = "n") %>%
      group_by(bin) %>%
      summarise(exp = mean(n, na.rm = TRUE), .groups = "drop")

    full_join(obs, exp, by = "bin") %>%
      mutate(obs   = replace_na(obs, 0),
             exp   = replace_na(exp, 0),
             ratio = obs / exp,
             celltype = celltype)
  })

  bind_rows(ratio) %>%
    mutate(bin      = as.numeric(bin),
           celltype = factor(celltype, levels = names(celltype_cols)))
}

plot_enrichment <- function(df, y_label) {
  ggplot(df, aes(x = bin, y = ratio, color = celltype, fill = celltype)) +
    stat_cor(aes(group = celltype), method = "pearson", size = 4,
             show.legend = FALSE, label.x.npc = "right", hjust = 1) +
    geom_smooth(method = "lm", se = TRUE, linewidth = 1.3, alpha = 0.08) +
    geom_point(size = 2) +
    scale_color_manual(values = celltype_cols) +
    scale_fill_manual(values = celltype_cols) +
    scale_x_continuous(breaks = 1:n_bins, labels = 1:n_bins) +
    labs(x = "Accessibility levels", y = y_label) +
    theme_classic(base_size = 14) +
    theme(legend.title = element_blank(),
          axis.text.x  = element_text(size = 12),
          axis.text.y  = element_text(size = 12))
}

################################################################################
# Fig 4j | sSNV enrichment in relation to chromatin accessibility
################################################################################

ggsave(file.path(output_path, "fig4j_snv_enrichment_accessibility.pdf"),
       plot = plot_enrichment(accessibility_enrichment("snv"), "sSNV enrichment (obs/exp)"),
       width = 5, height = 4)

################################################################################
# Fig 4j | sIndel enrichment in relation to chromatin accessibility
################################################################################

ggsave(file.path(output_path, "fig4j_indel_enrichment_accessibility.pdf"),
       plot = plot_enrichment(accessibility_enrichment("indel"), "sIndel enrichment (obs/exp)"),
       width = 5, height = 4)
