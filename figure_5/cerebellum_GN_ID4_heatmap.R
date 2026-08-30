################################################################################
# Fig 5 | ID4 signature landscape across regions and phenotypes
################################################################################

library(ComplexHeatmap)
library(circlize)
library(grid)

input_path  <- "data"
output_path <- "results"

NA_COL <- "#BDBDBD"

col_celltype <- c(`cortical neuron` = "#363636", `granule neuron` = "#c62f5a")
col_pheno    <- c(Control = "#808180", AD = "#4DBBD5", ALS = "#00A087",
                  FTD = "#3C5488", CTE = "#9B5C97")
col_region   <- c(CB = "#d4a373", CH = "#706993", V = "#9bc1bc",
                  BA4 = "#E64B35", BA6 = "#F39B7F", `BA9/46` = "#DC0000")
col_amp      <- c(MDA = "#7E6148", PTA = "#374E55")
col_id4      <- c(ID4_high = "#B2182B", ID4_normal = "#2166AC")
col_excess   <- colorRamp2(c(-50, 0, 50, 150, 400),
                           c("#2166AC", "#F7F7F7", "#FDDBC7", "#D6604D", "#67001F"))
col_exburden <- colorRamp2(c(-150, 0, 60, 200, 500),
                           c("#762A83", "#F7F7F7", "#D9F0D3", "#5AAE61", "#1B7837"))

################################################################################
# Signature matrix and cell annotation
################################################################################

si <- read.csv(file.path(input_path, "ID4_RF_labels.csv"), check.names = FALSE)
stopifnot(!any(duplicated(si$cell)), !any(is.na(si$final_label)))

sg <- as.matrix(read.csv(file.path(input_path, "mut_sig.csv"),
                         row.names = 1, check.names = FALSE))
M  <- sweep(sg, 2, pmax(colSums(sg), 1e-9), "/")

si$ID4_prop <- unname((sg["ID4", ] / pmax(colSums(sg), 1e-12))[match(si$cell, colnames(sg))])

tab <- si[, c("cell", "donor", "phenotype", "region2", "region", "celltype",
              "age", "indel_burden", "exp.indel_burden", "excess.indel_burden",
              "amp", "ID4_prop", "exp.ID4", "excess.ID4", "ID4.status",
              "RF_prob", "predict_label", "final_label")]
write.csv(tab, file.path(output_path, "ID4_heatmap_annotation_table.csv"), row.names = FALSE)

gap <- setdiff(tab$cell, colnames(M))
if (length(gap)) {
  M <- cbind(M, matrix(NA_real_, nrow(M), length(gap),
                       dimnames = list(rownames(M), gap)))
}

a <- tab[!is.na(tab$final_label), ]
a$region2     <- factor(a$region2, levels = c("Cerebellum", "Cortex"))
a$phenotype   <- factor(a$phenotype, levels = names(col_pheno))
a$final_label <- factor(a$final_label, levels = c("ID4_high", "ID4_normal"))

a <- a[order(a$region2, a$phenotype, as.integer(a$final_label),
             -a$excess.ID4, -a$ID4_prop, na.last = TRUE), ]
M <- M[, a$cell, drop = FALSE]
a$grp <- droplevels(interaction(a$region2, a$phenotype, sep = " / ", lex.order = TRUE))

################################################################################
# Proportion of ID4_high per region and phenotype
################################################################################

groups <- levels(a$grp)
summ <- data.frame(group  = groups,
                   n      = as.integer(table(a$grp)[groups]),
                   n_high = as.integer(tapply(a$final_label == "ID4_high", a$grp, sum)[groups]))
summ$pct_high <- 100 * summ$n_high / summ$n
ci <- t(mapply(function(x, n) if (n == 0) c(NA, NA) else binom.test(x, n)$conf.int * 100,
               summ$n_high, summ$n))
summ$ci_low <- ci[, 1]; summ$ci_high <- ci[, 2]
write.csv(summ, file.path(output_path, "ID4_proportion_by_group.csv"), row.names = FALSE)

pairs <- t(combn(length(groups), 2))
res <- do.call(rbind, lapply(seq_len(nrow(pairs)), function(i) {
  g1 <- groups[pairs[i, 1]]; g2 <- groups[pairs[i, 2]]
  x <- summ$n_high[match(c(g1, g2), summ$group)]
  n <- summ$n[match(c(g1, g2), summ$group)]
  data.frame(group1 = g1, group2 = g2,
             n1 = n[1], high1 = x[1], pct1 = 100 * x[1] / n[1],
             n2 = n[2], high2 = x[2], pct2 = 100 * x[2] / n[2],
             comparison = ifelse(sub(" /.*", "", g1) == sub(" /.*", "", g2),
                                 paste0("within ", sub(" /.*", "", g1)), "across region2"),
             p_prop_test = tryCatch(suppressWarnings(prop.test(x, n)$p.value),
                                    error = function(e) NA_real_),
             p_fisher = tryCatch(fisher.test(matrix(c(x, n - x), nrow = 2))$p.value,
                                 error = function(e) NA_real_))
}))
res$FDR_prop_test <- p.adjust(res$p_prop_test, method = "BH")
res$FDR_fisher    <- p.adjust(res$p_fisher,    method = "BH")
res <- res[order(res$p_fisher), ]
write.csv(res, file.path(output_path, "ID4_proportion_tests.csv"), row.names = FALSE)

################################################################################
# Fig 5 | Heatmap
################################################################################

lgd_base <- list(labels_gp = gpar(fontsize = 6),
                 title_gp = gpar(fontsize = 7, fontface = "bold"),
                 grid_height = unit(3, "mm"), grid_width = unit(3, "mm"))
anno_lgd <- setNames(rep(list(lgd_base), 8),
                     c("celltype", "phenotype", "region", "age", "amp",
                       "excess.indel_burden", "excess.ID4", "ID4 group"))
anno_lgd$excess.indel_burden <- c(lgd_base, list(
  at     = c(-150, 0, 150, 300, 500),
  labels = expression(-150, 0, 150, 300, phantom() >= 500)))
anno_lgd$excess.ID4 <- c(lgd_base, list(
  at     = c(-50, 0, 100, 250, 400),
  labels = expression(-50, 0, 100, 250, phantom() >= 400)))

ha <- HeatmapAnnotation(
  celltype = a$celltype, phenotype = a$phenotype, region = as.character(a$region),
  age = a$age, amp = a$amp, excess.indel_burden = a$excess.indel_burden,
  excess.ID4 = pmin(a$excess.ID4, 400),
  `ID4 group` = a$final_label,
  col = list(celltype = col_celltype, phenotype = col_pheno, region = col_region,
             amp = col_amp, `ID4 group` = col_id4,
             age = colorRamp2(range(a$age, na.rm = TRUE), c("#FFFFFF", "#542788")),
             excess.indel_burden = col_exburden,
             excess.ID4 = col_excess),
  na_col = NA_COL,
  annotation_name_gp = gpar(fontsize = 7), annotation_name_side = "left",
  simple_anno_size = unit(3.2, "mm"), gap = unit(0.5, "mm"),
  annotation_legend_param = anno_lgd
)

pct_cell <- 100 * as.numeric(tapply(a$final_label == "ID4_high", a$grp,
                                    mean)[as.character(a$grp)])
ba <- HeatmapAnnotation(
  `% ID4_high` = anno_barplot(pct_cell, ylim = c(0, 100), border = FALSE, bar_width = 1,
                              gp = gpar(fill = col_id4[["ID4_high"]], col = NA),
                              axis_param = list(gp = gpar(fontsize = 6), at = c(0, 50, 100)),
                              height = unit(14, "mm")),
  pct = anno_block(gp = gpar(fill = NA, col = NA), height = unit(7, "mm"),
                   panel_fun = function(index, nm) {
                     grid.text(sprintf("%.1f%%\nn=%d",
                                       100 * mean(a$final_label[index] == "ID4_high"),
                                       length(index)),
                               gp = gpar(fontsize = 6))
                   }),
  annotation_name_gp = gpar(fontsize = 7), annotation_name_side = "left",
  show_annotation_name = c(`% ID4_high` = TRUE, pct = FALSE),
  gap = unit(0.5, "mm")
)

slices  <- unique(data.frame(r = as.character(a$region2), p = as.character(a$phenotype)))
n_slice <- nrow(slices)
gap_vec <- rep(0.7, n_slice - 1)
gap_vec[which(slices$r[-n_slice] != slices$r[-1])] <- 1.4

ht <- Heatmap(
  M, name = "signature\nproportion",
  col = colorRamp2(c(0, 0.5, 1), c("#FCFCFB", "#6DA7EC", "#0D366B")),
  na_col = NA_COL, top_annotation = ha, bottom_annotation = ba,
  column_split = a[, c("region2", "phenotype")],
  cluster_columns = FALSE, cluster_column_slices = FALSE, show_column_names = FALSE,
  cluster_rows = TRUE, show_row_dend = FALSE, row_names_gp = gpar(fontsize = 7),
  column_title_gp = gpar(fontsize = 7, fontface = "bold"), column_title_rot = 90,
  column_gap = unit(gap_vec, "mm"), border = FALSE,
  heatmap_legend_param = list(
    labels_gp = gpar(fontsize = 6), title_gp = gpar(fontsize = 7, fontface = "bold"),
    legend_height = unit(15, "mm"), grid_width = unit(3, "mm")),
  use_raster = TRUE, raster_quality = 4
)

lgd_na <- Legend(labels = "NA (not available)", title = "missing",
                 legend_gp = gpar(fill = NA_COL),
                 labels_gp = gpar(fontsize = 6),
                 title_gp = gpar(fontsize = 7, fontface = "bold"),
                 grid_height = unit(3, "mm"), grid_width = unit(3, "mm"))

pdf(file.path(output_path, "fig5e_ID4_GN_heatmap.pdf"), width = 328 / 25.4, height = 185 / 25.4)
draw(ht, merge_legend = TRUE, heatmap_legend_side = "right",
     annotation_legend_side = "right", annotation_legend_list = list(lgd_na))
dev.off()
