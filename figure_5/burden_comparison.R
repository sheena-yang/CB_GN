################################################################################
# Fig 5 | Age-adjusted sSNV and sIndel burden in disease versus control GNs
################################################################################

library(dplyr)
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggpubr)
library(rstatix)

input_path  <- "data"
output_path <- "results"

clinical_levels <- c("Control", "AD", "ALS", "FTD")
clinical_cols   <- c("Control" = "#808180",
                     "AD"      = "#4DBBD5",
                     "ALS"     = "#00A087",
                     "FTD"     = "#3C5488")

clinical_pairs <- list(c("Control", "AD"), c("Control", "ALS"), c("Control", "FTD"),
                       c("AD", "ALS"), c("AD", "FTD"), c("ALS", "FTD"))

################################################################################
# Control and disease GNs
################################################################################

clinical_burden <- function(burden_file) {
  df_burden <- read.csv(file.path(input_path, burden_file))

  control_med3 <- df_burden %>%
    filter(clinical == "Control") %>%
    group_by(donor, celltype) %>%
    mutate(med_burden  = median(burden, na.rm = TRUE),
           dist_to_med = abs(burden - med_burden)) %>%
    arrange(dist_to_med, .by_group = TRUE) %>%
    slice_head(n = 3) %>%
    ungroup() %>%
    select(-med_burden, -dist_to_med)

  bind_rows(control_med3, df_burden %>% filter(clinical != "Control")) %>%
    mutate(clinical = factor(clinical, levels = clinical_levels))
}

plot_obs_exp <- function(disease, model, y_label, y_positions) {

  cf <- fixef(model)
  disease_plot <- disease %>%
    mutate(exp_burden = cf["(Intercept)"] + cf["age"] * age,
           obs_exp    = burden / exp_burden)

  stat_test <- disease_plot %>%
    wilcox_test(obs_exp ~ clinical, comparisons = clinical_pairs) %>%
    adjust_pvalue(method = "BH") %>%
    add_significance("p.adj") %>%
    mutate(y.position = y_positions)

  ggplot(disease_plot, aes(x = clinical, y = obs_exp, fill = clinical)) +
    geom_boxplot(width = 0.6, outlier.shape = NA, alpha = 0.7) +
    geom_point(position = position_jitter(width = 0.12, height = 0), size = 1, alpha = 0.8) +
    scale_fill_manual(values = clinical_cols) +
    stat_pvalue_manual(stat_test, label = "p.adj", tip.length = 0.01, inherit.aes = FALSE) +
    labs(x = NULL, y = y_label) +
    theme_classic(base_size = 14) +
    theme(legend.position = "none")
}

################################################################################
# Fig 5b | SNV burden
################################################################################

disease_snv <- clinical_burden("snv_burden_disease_control.csv")

model_clinical_snv <- lmer(burden ~ age + clinical + (1 | donor), data = disease_snv)
summary(model_clinical_snv)

ggsave(file.path(output_path, "fig5b_clinical_snv_burden_comparison.pdf"),
       plot = plot_obs_exp(disease_snv, model_clinical_snv,
                           "sSNV burden (obs / exp)",
                           c(2, 2.2, 2.4, 2.6, 2.8, 3.0)),
       width = 4, height = 4)

################################################################################
# Fig 5c | Indel burden
################################################################################

disease_indel <- clinical_burden("indel_burden_disease_control.csv")

model_clinical_indel <- lmer(burden ~ age + clinical + (1 | donor), data = disease_indel)
summary(model_clinical_indel)

ggsave(file.path(output_path, "fig5c_clinical_indel_burden_comparison.pdf"),
       plot = plot_obs_exp(disease_indel, model_clinical_indel,
                           "sIndel burden (obs / exp)",
                           c(4, 4.3, 4.6, 4.9, 5.2, 5.5)),
       width = 4, height = 4)
