################################################################################
# Fig 1 | Somatic indel burden versus age across cell types
################################################################################

library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)

input_path  <- "data"
output_path <- "results"

# granule neuron
df_burden <- read.csv(file.path(input_path, "indel_burden.csv"))

# published cortical neurons and OLs
burden_cortical <- read.csv(file.path(input_path, "Ganz_Cell2024_indel_burden.csv"))
burden_cortical <- burden_cortical[burden_cortical$outlier == "NORMAL", ]

burden_m <- rbind(df_burden, burden_cortical)
burden_m$celltype <- factor(burden_m$celltype,
                         levels = c("granule_neuron", "cortical_neuron", "cortical_OL"))

################################################################################
# Linear mixed model
################################################################################

# three cells closest to the median burden per donor and cell type
burden_m_med3 <- burden_m %>%
  group_by(donor, celltype) %>%
  mutate(med_burden  = median(burden, na.rm = TRUE),
         dist_to_med = abs(burden - med_burden)) %>%
  arrange(dist_to_med, .by_group = TRUE) %>%
  slice_head(n = 3) %>%
  ungroup() %>%
  select(-med_burden, -dist_to_med)

model <- lmer(burden ~ age * celltype + (1 | donor), data = burden_m_med3)
summary(model)

slopes <- emtrends(model, ~ celltype, var = "age")
confint(slopes)

################################################################################
# Fig 1e | Indel burden versus age
################################################################################

cf <- fixef(model)

line_df <- data.frame(
  celltype = c("granule_neuron", "cortical_neuron", "cortical_OL"),
  intercept = c(cf["(Intercept)"],
                cf["(Intercept)"] + cf["celltypecortical_neuron"],
                cf["(Intercept)"] + cf["celltypecortical_OL"]),
  slope = c(cf["age"],
            cf["age"] + cf["age:celltypecortical_neuron"],
            cf["age"] + cf["age:celltypecortical_OL"])
)

age_grid <- seq(min(burden_m$age, na.rm = TRUE),
                max(burden_m$age, na.rm = TRUE), length.out = 100)

pred_df <- crossing(age = age_grid, celltype = line_df$celltype) %>%
  left_join(line_df, by = "celltype") %>%
  mutate(pred  = intercept + slope * age,
         celltype = factor(celltype, levels = c("granule_neuron", "cortical_neuron", "cortical_OL")))

p <- ggplot(burden_m, aes(x = age, y = burden)) +
  geom_line(data = pred_df, aes(x = age, y = pred, color = celltype), linewidth = 1.2) +
  geom_point(data = burden_m[burden_m$celltype == "granule_neuron", ],
             aes(color = celltype), size = 2, alpha = 0.95) +
  scale_x_continuous(breaks = c(0, 20, 40, 60, 80, 100)) +
  scale_color_manual(values = c("granule_neuron"  = "#c62f5a",
                                "cortical_neuron" = "#363636",
                                "cortical_OL"     = "#ffbc42"), name = "Cell type") +
  labs(x = "Age", y = "sIndels per cell") +
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

ggsave(file.path(output_path, "fig1e_indel_burden_comparison.pdf"), plot = p,
       width = 6.7, height = 5)
