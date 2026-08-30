################################################################################
# Fig 3 | COSMIC signature refitting of single-cell indel spectra
################################################################################

library(MutationalPatterns)
library(pracma)
library(dplyr)
library(tidyr)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(lme4)
library(lmerTest)
library(ggpubr)
library(rstatix)

input_path  <- "data"
output_path <- "results"

celltype_levels <- c("granule_neuron", "cortical_neuron", "cortical_OL")
celltype_cols   <- c("granule_neuron"  = "#c62f5a",
                     "cortical_neuron" = "#363636",
                     "cortical_OL"     = "#ffbc42")

################################################################################
# Mutation matrix and metadata
################################################################################

mut_mat_indel <- read.table(file.path(input_path, "indel_83_matrix.txt"),
                            header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
mut_mat_indel <- mut_mat_indel + 0.0001

cell_meta          <- read.csv(file.path(input_path, "indel_burden.csv"))
cell_meta          <- cell_meta[cell_meta$burden > 0, ]
cell_meta$celltype <- factor(cell_meta$celltype, levels = celltype_levels)

cell_order <- cell_meta$cell_id[order(cell_meta$celltype, cell_meta$age, cell_meta$cell_id)]
cell_meta$ordered_cell <- factor(cell_meta$cell_id, levels = cell_order)

mut_mat_indel <- mut_mat_indel[, intersect(cell_meta$cell_id, colnames(mut_mat_indel))]
cell_meta     <- cell_meta[match(colnames(mut_mat_indel), cell_meta$cell_id), ]

################################################################################
# Signature decomposition analysis
################################################################################

# best-fitting signature, by non-negative least squares
sigs2fit <- get_known_signatures(muttype = "indel", incl_poss_artifacts = TRUE)
for (i in seq_len(ncol(sigs2fit))) {
  sigs2fit[, i] <- sigs2fit[, i] / sum(sigs2fit[, i])
}

sig_used   <- rep(FALSE, ncol(sigs2fit))
prev_resid <- sum(mut_mat_indel^2)

next_resids <- sapply(which(!sig_used), function(i) {
  do.call(sum, lapply(seq_len(ncol(mut_mat_indel)), function(j) {
    lsqnonneg(sigs2fit[, i, drop = FALSE], mut_mat_indel[, j])$resid.norm
  }))
})
sig_used[!sig_used][which.min(next_resids)] <- TRUE
sig_added  <- colnames(sigs2fit)[sig_used]
next_resid <- min(next_resids)
print(sig_added)

fit_resid    <- rep(NA, ncol(sigs2fit))
fit_resid[1] <- next_resid
i <- 2
while ((prev_resid - next_resid) / prev_resid > 0.01) {
  next_resids <- sapply(which(!sig_used), function(i) {
    this_sig_used    <- sig_used
    this_sig_used[i] <- TRUE
    do.call(sum, lapply(seq_len(ncol(mut_mat_indel)), function(j) {
      lsqnonneg(sigs2fit[, this_sig_used, drop = FALSE], mut_mat_indel[, j])$resid.norm
    }))
  })
  sig_used[!sig_used][which.min(next_resids)] <- TRUE
  sig_added <- c(sig_added, setdiff(colnames(sigs2fit)[sig_used], sig_added))
  print(sig_added)
  prev_resid   <- next_resid
  next_resid   <- min(next_resids)
  fit_resid[i] <- next_resid
  i <- i + 1
}

fit_resid_improvement <- -fit_resid[2:length(fit_resid)] + fit_resid[1:(length(fit_resid) - 1)]

use_cosmic_sigs <- sig_added[seq_len(sum(fit_resid_improvement > 1500, na.rm = TRUE) + 1)]

cell_cosmic_fit <- fit_to_signatures(mut_mat_indel, sigs2fit[, use_cosmic_sigs])

# scale contributions so that each cell sums to its estimated sSNV burden
contribution <- cell_cosmic_fit$contribution
for (i in seq_len(ncol(contribution))) {
  contribution[, i] <- contribution[, i] *
    cell_meta$burden[match(colnames(contribution)[i], cell_meta$cell_id)] /
    sum(cell_cosmic_fit$reconstructed[, i])
}

contribution <- contribution[order(rowSums(contribution), decreasing = FALSE), ]

cell_order2 <- order(
  cell_meta$celltype[match(colnames(contribution), cell_meta$cell_id)],
  cell_meta$age[match(colnames(contribution), cell_meta$cell_id)],
  cell_meta$donor[match(colnames(contribution), cell_meta$cell_id)],
  cell_meta$cell_id[match(colnames(contribution), cell_meta$cell_id)])
contribution <- contribution[, cell_order2]

contribution_df         <- as.data.frame(t(contribution))
contribution_df$cell_id <- rownames(contribution_df)
contribution_df         <- merge(contribution_df, cell_meta)
contribution_long       <- melt(contribution_df, id.vars = colnames(cell_meta))

################################################################################
# Fig 3c | Signature contribution per cell
################################################################################

p_spectrum <- ggplot(contribution_long, aes(ordered_cell, value, fill = variable)) +
  geom_col() +
  facet_grid(~ celltype, scales = "free_x", space = "free_x") +
  scale_y_continuous(limits = c(0, 370), expand = c(0, 0)) +
  labs(x = "", y = "Number of mutations") +
  theme_classic() +
  theme(panel.grid.minor.x = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.ticks.x       = element_blank(),
        axis.text.x        = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.position    = "right",
        legend.title       = element_blank())

ggsave(file.path(output_path, "fig3c_sindel_mutation_spectrum.pdf"), plot = p_spectrum,
       width = 24, height = 5)

################################################################################
# Fig 3d | Signature contribution versus age
################################################################################

age_breaks  <- c(0, 20, 40, 60, 80, 100)
panel_order <- c(1, 4, 5, 6, 2, 3, 8)

sbs_plots <- lapply(unique(contribution_long$variable)[8:1], function(sbs) {

  data_raw <- contribution_long %>%
    filter(variable == sbs) %>%
    mutate(celltype = factor(celltype, levels = celltype_levels))

  model <- lmer(value ~ age * celltype + (1 | donor), data = data_raw)
  print(sbs)
  print(summary(model))

  cf      <- fixef(model)
  line_df <- data.frame(
    celltype = celltype_levels,
    intercept = c(cf["(Intercept)"],
                  cf["(Intercept)"] + cf["celltypecortical_neuron"],
                  cf["(Intercept)"] + cf["celltypecortical_OL"]),
    slope = c(cf["age"],
              cf["age"] + cf["age:celltypecortical_neuron"],
              cf["age"] + cf["age:celltypecortical_OL"])
  )

  age_grid <- seq(min(data_raw$age, na.rm = TRUE),
                  max(data_raw$age, na.rm = TRUE), length.out = 100)

  pred_df <- crossing(age = age_grid, celltype = celltype_levels) %>%
    left_join(line_df, by = "celltype") %>%
    mutate(pred     = intercept + slope * age,
           celltype = factor(celltype, levels = celltype_levels))

  p <- ggplot(data_raw, aes(x = age, y = value)) +
    geom_line(data = pred_df, aes(x = age, y = pred, color = celltype), linewidth = 1.2) +
    geom_point(aes(color = celltype), size = 1.8, alpha = 0.8) +
    scale_x_continuous(breaks = age_breaks, labels = age_breaks) +
    scale_color_manual(values = celltype_cols, name = "Cell type") +
    labs(x = "Age", y = sbs) +
    theme_minimal() +
    theme(panel.grid        = element_blank(),
          panel.border      = element_blank(),
          axis.line         = element_line(color = "black"),
          axis.ticks        = element_line(color = "black"),
          axis.ticks.length = unit(0.2, "cm"),
          text              = element_text(size = 15),
          axis.title        = element_text(size = 15),
          axis.text         = element_text(size = 15),
          legend.text       = element_text(size = 12),
          legend.title      = element_text(size = 13),
          plot.title        = element_text(size = 18, face = "bold", hjust = 0.5))

  if (sbs != "ID9") {
    p <- p + guides(color = "none")
  } else {
    p <- p + theme(legend.position = c(0.22, 0.9), legend.title = element_blank())
  }
  p
})

ggsave(file.path(output_path, "fig3d_cosmic_age_burden_indel.pdf"),
       plot = marrangeGrob(sbs_plots[panel_order], nrow = 1, ncol = 7, top = ""),
       width = 35, height = 5)

################################################################################
# Fig 3e | Excess signature burden relative to the granule neuron trajectory
################################################################################

sbs_residual_plots <- lapply(unique(contribution_long$variable)[8:1], function(sbs) {

  data_raw <- contribution_long %>%
    filter(variable == sbs) %>%
    mutate(celltype = factor(celltype, levels = celltype_levels)) %>%
    filter(!is.na(celltype), !is.na(age), !is.na(value))

  # aging trajectory fitted on granule neurons only
  model_gc <- lmer(value ~ age + (1 | donor),
                   data = data_raw %>% filter(celltype == "granule_neuron"))
  print(sbs)
  print(summary(model_gc))

  data_plot <- data_raw %>%
    mutate(gc_expected = predict(model_gc, newdata = data_raw, re.form = NA),
           gc_resid    = value - gc_expected)

  stat_test <- data_plot %>%
    wilcox_test(gc_resid ~ celltype,
                comparisons = list(c("granule_neuron", "cortical_neuron"),
                                   c("granule_neuron", "cortical_OL"))) %>%
    add_significance("p") %>%
    mutate(p_label    = paste0("p = ", signif(p, 3)),
           y.position = max(data_plot$gc_resid, na.rm = TRUE) * c(1.08, 1.22))

  ggplot(data_plot, aes(x = celltype, y = gc_resid)) +
    geom_violin(width = 0.6, alpha = 0.8, scale = "width", trim = TRUE) +
    geom_boxplot(aes(colour = celltype, fill = celltype),
                 alpha = 0.5, size = 1, width = 0.2) +
    geom_jitter(aes(colour = celltype), alpha = 0.3, size = 1) +
    stat_pvalue_manual(stat_test, label = "p_label", tip.length = 0.01, inherit.aes = FALSE) +
    scale_fill_manual(values = celltype_cols) +
    scale_color_manual(values = celltype_cols) +
    labs(x = NULL, y = paste0(sbs, " Excess sIndels per cell")) +
    theme_minimal() +
    theme(panel.grid      = element_blank(),
          panel.border    = element_blank(),
          axis.line       = element_line(color = "black"),
          axis.ticks      = element_line(color = "black"),
          text            = element_text(size = 15),
          axis.title      = element_text(size = 15),
          axis.text       = element_text(size = 15),
          legend.text     = element_text(size = 12),
          legend.title    = element_text(size = 13),
          legend.position = "none",
          plot.title      = element_text(size = 18, face = "bold", hjust = 0.5))
})

ggsave(file.path(output_path, "fig3e_cosmic_burden_residual_indel.pdf"),
       plot = marrangeGrob(sbs_residual_plots[panel_order], nrow = 1, ncol = 7, top = ""),
       width = 35, height = 5)
