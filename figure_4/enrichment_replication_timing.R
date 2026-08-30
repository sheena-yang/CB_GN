################################################################################
# Fig 4 | sSNV and sIndel enrichment in relation to replication timing
################################################################################

library(data.table)
library(future.apply)
library(GenomicRanges)
library(rtracklayer)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ggpubr)

plan(multicore, workers = 20)

input_path  <- "data"
output_path <- "results"

n_bins <- 5
n_perm <- 1000

celltype_cols <- c("granule_neuron"  = "#c62f5a",
                   "cortical_neuron" = "#363636",
                   "cortical_OL"     = "#ffbc42")

################################################################################
# Replication timing bins from Repli-seq
################################################################################

bw_files <- list.files(file.path(input_path, "RepliSeq"), pattern = "Rep1.bigWig$", full.names = TRUE)

rep_bins <- bind_rows(lapply(bw_files, function(bw) {
  rt <- import.bw(bw)
  data.frame(chr   = as.character(seqnames(rt)),
             start = start(rt),
             end   = end(rt),
             bin   = (n_bins + 1) - ntile(rt$score, n = n_bins))
}))

rep_bins <- rep_bins %>%
  group_by(chr, start, end) %>%
  summarise(bin = as.integer(round(median(bin, na.rm = TRUE))), .groups = "drop")

rep_gr <- GRanges(seqnames = rep_bins$chr,
                  ranges = IRanges(rep_bins$start, rep_bins$end),
                  bin = rep_bins$bin)

################################################################################
# Observed and permuted mutations per replication timing bin
################################################################################

assign_bin <- function(mut_df) {
  mut_gr <- GRanges(seqnames = paste0("chr", mut_df$chr),
                    ranges = IRanges(mut_df$start, mut_df$end))
  hits <- findOverlaps(mut_gr, rep_gr, select = "first")
  mut_df$bin <- NA_integer_
  mut_df$bin[!is.na(hits)] <- mcols(rep_gr)$bin[hits[!is.na(hits)]]
  mut_df
}

read_observed <- function(annovar_path, muttype) {
  files <- list.files(file.path(input_path, annovar_path),
                      pattern = paste0("pass_", muttype, ".hg19_multianno.txt$"),
                      full.names = TRUE, recursive = TRUE)
  bind_rows(lapply(files, function(f) {
    fread(f, select = c("Chr", "Start", "End"), col.names = c("chr", "start", "end"))
  }))
}

read_permuted <- function(perm_path, muttype) {
  files <- list.files(file.path(input_path, perm_path),
                      pattern = paste0("perms_", muttype, ".hg19_multianno.txt$"),
                      full.names = TRUE, recursive = TRUE)
  bind_rows(future_lapply(files, function(f) {
    dt <- fread(f, select = c("Chr", "Start", "End"), col.names = c("chr", "start", "end"))
    stopifnot(nrow(dt) %% n_perm == 0)
    dt[, perm.id := rep(seq_len(n_perm), each = nrow(dt) / n_perm)]
    dt
  }))
}

replication_enrichment <- function(muttype) {

  ratio <- lapply(names(celltype_cols), function(celltype) {

    observed <- assign_bin(read_observed(file.path("annovar", celltype), muttype))
    permuted <- assign_bin(read_permuted(file.path("perm", celltype), muttype))

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
    labs(x = "Replication timing", y = y_label) +
    theme_classic(base_size = 14) +
    theme(legend.title = element_blank(),
          axis.text.x  = element_text(size = 12),
          axis.text.y  = element_text(size = 12))
}

################################################################################
# Fig 4i | sSNV enrichment in relation to replication timing
################################################################################

ggsave(file.path(output_path, "fig4i_snv_enrichment_replication_timing.pdf"),
       plot = plot_enrichment(replication_enrichment("snv"), "sSNV enrichment (obs/exp)"),
       width = 5, height = 4)

################################################################################
# Fig 4i | sIndel enrichment in relation to replication timing
################################################################################

ggsave(file.path(output_path, "fig4i_indel_enrichment_replication_timing.pdf"),
       plot = plot_enrichment(replication_enrichment("indel"), "sIndel enrichment (obs/exp)"),
       width = 5, height = 4)
