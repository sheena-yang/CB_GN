################################################################################
# Fig 4 | Predicted functional impact of somatic mutations
################################################################################

library(data.table)
library(ggplot2)
library(ggpubr)

input_path  <- "data"
output_path <- "results"

impact_levels <- c("HIGH", "MODERATE", "LOW", "MODIFIER")

celltype_cols <- c("granule_neuron"  = "#c62f5a",
                   "cortical_neuron" = "#363636",
                   "cortical_OL"     = "#ffbc42")

donor_gn       <- function(cell) substr(cell, 1, 4)
donor_cortical <- function(cell) sub("[-_].*", "", cell)

# impact of the first snpEff annotation in the INFO field
first_impact <- function(info) {
  ann <- ifelse(grepl("ANN=", info, fixed = TRUE), sub("^.*?ANN=", "", info), NA_character_)
  ann <- sub(",.*$", "", sub(";.*$", "", ann))
  vapply(strsplit(ann, "|", fixed = TRUE),
         function(x) if (length(x) >= 3) x[3] else NA_character_,
         character(1))
}

################################################################################
# Impact level per cell
################################################################################

count_impact <- function(snpeff_path, muttype, donor_fun) {
  files <- list.files(snpeff_path, pattern = paste0("pass_", muttype, ".hg19_snpeff.vcf$"),
                      full.names = TRUE, recursive = TRUE)
  dt_list <- lapply(files, function(f) {
    vcf  <- fread(f, skip = "#CHROM", header = TRUE)
    cell <- basename(dirname(f))
    dt   <- data.table(impact = first_impact(vcf$INFO))[impact %in% impact_levels]
    dt[, .N, by = impact][, `:=`(cell = cell, donor = donor_fun(cell))]
  })
  dcast(rbindlist(dt_list, fill = TRUE), cell + donor ~ impact, value.var = "N", fill = 0)
}

# percent of mutations per impact level, pooled within donor and cell type
impact_percent <- function(muttype) {

  observed <- list(
    granule_neuron  = count_impact(file.path(input_path, "annovar/granule_neuron"), muttype, donor_gn),
    cortical_neuron = count_impact(file.path(input_path, "annovar/cortical_neuron"), muttype, donor_cortical),
    cortical_OL     = count_impact(file.path(input_path, "annovar/cortical_OL"), muttype, donor_cortical)
  )
  for (ct in names(observed)) observed[[ct]][, celltype := ct]

  donor_dt <- rbindlist(observed, fill = TRUE)[, lapply(.SD, sum),
                                               by = .(donor, celltype), .SDcols = impact_levels]
  donor_dt[, total := rowSums(.SD), .SDcols = impact_levels]
  for (level in c("HIGH", "MODERATE", "LOW")) {
    donor_dt[[level]] <- donor_dt[[level]] / donor_dt$total * 100
  }

  df <- melt(donor_dt[, .(donor, celltype, HIGH, MODERATE, LOW)],
             id.vars = c("donor", "celltype"),
             variable.name = "impact", value.name = "percent")
  df[, impact := factor(impact, levels = c("HIGH", "MODERATE", "LOW"))]
  df
}

plot_impact <- function(df, y_label) {
  ggplot(df, aes(x = celltype, y = percent, color = celltype)) +
    geom_boxplot(outlier.shape = NA, width = 0.6) +
    stat_compare_means(comparisons = list(c("granule_neuron", "cortical_neuron"),
                                          c("granule_neuron", "cortical_OL")),
                       method = "wilcox.test", label = "p.signif",
                       size = 4, tip.length = 0.01) +
    facet_wrap(~ impact, nrow = 1) +
    scale_color_manual(values = celltype_cols) +
    labs(x = "Cell type", y = y_label, color = "Cell type") +
    theme_classic(base_size = 14) +
    theme(axis.text.x      = element_text(angle = 45, hjust = 1),
          strip.background = element_blank(),
          strip.text       = element_text(size = 12),
          legend.position  = "none")
}

################################################################################
# Fig 4e | SNV functional impact
################################################################################

ggsave(file.path(output_path, "fig4e_snv_functional_impact.pdf"),
       plot = plot_impact(impact_percent("snv"), "Percent of sSNVs (%)"),
       width = 6, height = 4)

################################################################################
# Fig 4f | Indel functional impact
################################################################################

ggsave(file.path(output_path, "fig4f_indel_functional_impact.pdf"),
       plot = plot_impact(impact_percent("indel"), "Percent of sIndels (%)"),
       width = 6, height = 4)
