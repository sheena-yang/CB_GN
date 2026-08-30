################################################################################
# Fig 5 | ID4 status of granule neurons predicted from ID83 spectra
################################################################################

library(randomForest)
library(fastshap)
library(shapviz)
library(doParallel)
library(ggplot2)
library(patchwork)

input_path  <- "data"
output_path <- "results"

ntree     <- 1000
nodesize  <- 5
threshold <- 0.50
n_sim     <- 100

BASE <- 7
C1   <- "#2a78d6"
C2   <- "#eb6834"
SEQ_LO <- "#008bfb"
SEQ_HI <- "#ff0051"
INK  <- "#000000"
MUT  <- "#898781"
GRID <- "#e1e0d9"

theme_j <- function(bs = BASE) {
  theme_classic(base_size = bs) +
    theme(axis.line         = element_line(colour = INK, linewidth = 0.25),
          axis.ticks        = element_line(colour = INK, linewidth = 0.25),
          axis.ticks.length = unit(1.4, "pt"),
          axis.text         = element_text(colour = INK),
          axis.title        = element_text(colour = INK),
          plot.title        = element_text(colour = INK, face = "bold", size = bs * 1.15),
          plot.caption      = element_text(colour = MUT, size = bs * 0.85, hjust = 0),
          legend.title      = element_text(colour = INK, size = bs * 0.9),
          legend.text       = element_text(colour = INK, size = bs * 0.9),
          plot.margin       = margin(4, 5, 3, 3))
}

################################################################################
# Features and labels
################################################################################

mut <- as.matrix(read.csv(file.path(input_path, "mut_mat.csv"),
                          row.names = 1, check.names = FALSE))

X <- t(mut)
X <- X[, apply(X, 2, function(z) length(unique(z)) > 1), drop = FALSE]
channels <- colnames(X)
colnames(X) <- make.names(channels, unique = TRUE)

X <- X / pmax(rowSums(X), 1e-9)

si <- read.csv(file.path(input_path, "sample.info.csv"), check.names = FALSE)
X  <- X[si$cell[si$cell %in% rownames(X)], , drop = FALSE]
Xdf <- as.data.frame(X)

y     <- factor(si$ID4.status[match(rownames(Xdf), si$cell)],
                levels = c("ID4_normal", "ID4_high"))
train <- !is.na(y)
X_tr  <- Xdf[train, , drop = FALSE]
y_tr  <- droplevels(y[train])
donor <- si$donor[match(rownames(Xdf), si$cell)][train]

################################################################################
# Random forest and ID4 status prediction
################################################################################

fit_rf <- function(Xd, yd) {
  n_min <- min(table(yd))
  randomForest(x = Xd, y = yd, ntree = ntree, nodesize = nodesize,
               strata = yd, sampsize = c(n_min, n_min), importance = TRUE)
}

# case-held-out 5-fold cross-validation: no donor spans train and test
set.seed(99)
donors <- unique(donor)
fold   <- setNames(sample(rep_len(1:5, length(donors))), donors)
p_cv   <- rep(NA_real_, sum(train))
for (i in 1:5) {
  test <- fold[donor] == i
  if (all(test) || length(unique(y_tr[!test])) < 2) next
  m <- fit_rf(X_tr[!test, , drop = FALSE], droplevels(y_tr[!test]))
  p_cv[test] <- predict(m, newdata = X_tr[test, , drop = FALSE],
                        type = "prob")[, "ID4_high"]
}

set.seed(20260807)
rf    <- fit_rf(X_tr, y_tr)
p_oob <- rf$votes[, "ID4_high"]

# out-of-bag vote for training cells, forest prediction for the unlabelled ones
prob <- setNames(rep(NA_real_, nrow(Xdf)), rownames(Xdf))
prob[train]  <- p_oob
prob[!train] <- predict(rf, newdata = Xdf[!train, , drop = FALSE], type = "prob")[, "ID4_high"]

si$RF_prob       <- round(as.numeric(prob[si$cell]), 4)
si$predict_label <- ifelse(is.na(si$RF_prob), NA_character_,
                           ifelse(si$RF_prob >= threshold, "ID4_high", "ID4_normal"))
si$final_label   <- ifelse(!is.na(si$ID4.status), si$ID4.status, si$predict_label)

write.csv(si, file.path(output_path, "ID4_RF_labels.csv"), row.names = FALSE, na = "NA")

################################################################################
# Fig 5 | Model evaluation
################################################################################

roc_pr <- function(score, truth, pos = "ID4_high") {
  o <- order(score, decreasing = TRUE)
  s <- score[o]; t <- truth[o] == pos
  tp <- cumsum(t); fp <- cumsum(!t); P <- sum(t); N <- sum(!t)
  keep <- !duplicated(s, fromLast = TRUE)
  list(roc = data.frame(fpr = c(0, (fp / N)[keep]), tpr = c(0, (tp / P)[keep])),
       pr  = data.frame(recall = (tp / P)[keep], precision = (tp / (tp + fp))[keep]),
       auc = (sum(rank(score)[truth == pos]) - P * (P + 1) / 2) / (P * N),
       prevalence = P / (P + N))
}
auprc <- function(R) sum(diff(c(0, R$pr$recall)) * R$pr$precision)

scored <- !is.na(p_cv)
R_oob  <- roc_pr(p_oob, y_tr)
R_cv   <- roc_pr(p_cv[scored], y_tr[scored])

src_levels <- c("Out-of-bag", "Case-held-out CV")
roc <- rbind(data.frame(R_oob$roc, src = src_levels[1]),
             data.frame(R_cv$roc,  src = src_levels[2]))
pr  <- rbind(data.frame(R_oob$pr,  src = src_levels[1]),
             data.frame(R_cv$pr,   src = src_levels[2]))
roc$src <- factor(roc$src, src_levels)
pr$src  <- factor(pr$src,  src_levels)

p_roc <- ggplot(roc, aes(fpr, tpr, colour = src)) +
  geom_abline(slope = 1, intercept = 0, colour = GRID, linewidth = 0.3) +
  geom_step(linewidth = 0.5, direction = "hv") +
  scale_colour_manual(values = setNames(c(C1, C2), src_levels)) +
  annotate("text", x = 0.97, y = 0.30, hjust = 1, size = BASE / .pt,
           colour = C1, label = sprintf("Out-of-bag  AUROC = %.3f", R_oob$auc)) +
  annotate("text", x = 0.97, y = 0.21, hjust = 1, size = BASE / .pt,
           colour = C2, label = sprintf("Case-held-out CV  AUROC = %.3f", R_cv$auc)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(title = "a  ROC", x = "1 - specificity", y = "Sensitivity") +
  theme_j() + theme(legend.position = "none")

p_pr <- ggplot(pr, aes(recall, precision, colour = src)) +
  geom_hline(yintercept = R_oob$prevalence, colour = GRID, linewidth = 0.3) +
  geom_step(linewidth = 0.5, direction = "vh") +
  scale_colour_manual(values = setNames(c(C1, C2), src_levels)) +
  annotate("text", x = 0.97, y = R_oob$prevalence + 0.035, hjust = 1,
           size = BASE / .pt * 0.9, colour = MUT,
           label = sprintf("prevalence = %.3f", R_oob$prevalence)) +
  annotate("text", x = 0.03, y = 0.13, hjust = 0, size = BASE / .pt,
           colour = C1, label = sprintf("Out-of-bag  AUPRC = %.3f", auprc(R_oob))) +
  annotate("text", x = 0.03, y = 0.04, hjust = 0, size = BASE / .pt,
           colour = C2, label = sprintf("Case-held-out CV  AUPRC = %.3f", auprc(R_cv))) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(title = "b  Precision-recall", x = "Recall", y = "Precision") +
  theme_j() + theme(legend.position = "none")

ggsave(file.path(output_path, "RF_model_evaluation.pdf"), p_roc | p_pr,
       width = 165, height = 75, units = "mm")

write.csv(rbind(data.frame(metric = "AUROC", OOB = R_oob$auc,      donor_CV = R_cv$auc),
                data.frame(metric = "AUPRC", OOB = auprc(R_oob),   donor_CV = auprc(R_cv))),
          file.path(output_path, "ID4_RF_evaluation_metrics.csv"), row.names = FALSE)

################################################################################
# Fig 5 | SHAP explanation
################################################################################

pfun     <- function(object, newdata) predict(object, newdata = newdata, type = "prob")[, "ID4_high"]
baseline <- mean(pfun(rf, X_tr))

n_cores <- max(1, min(5, parallel::detectCores() - 1))
cl <- parallel::makeCluster(n_cores, type = "FORK")
registerDoParallel(cl)
S <- explain(rf, X = X_tr, newdata = Xdf, pred_wrapper = pfun,
             nsim = n_sim, adjust = TRUE, baseline = baseline,
             shap_only = TRUE, parallel = TRUE)
parallel::stopCluster(cl)
registerDoSEQ()

S <- as.matrix(S)
rownames(S) <- rownames(Xdf)
colnames(S) <- channels

Xp <- Xdf
colnames(Xp) <- channels

# proportions are right-skewed; scale to percentage points so the colour ramp
# spreads over its full range
sv_col <- shapviz(S, X = as.data.frame(log1p(100 * as.matrix(Xp)), check.names = FALSE),
                  baseline = baseline)

caption <- paste0("fastshap Monte-Carlo Shapley values (nsim = ", n_sim,
                  ") for P(ID4_high); ", nrow(S), " cells.")

p <- sv_importance(sv_col, kind = "beeswarm", max_display = 12,
                   bee_width = 0.42, bee_adjust = 0.6, size = 0.25) +
  scale_colour_gradient(low = SEQ_LO, high = SEQ_HI, limits = c(0, 1),
                        breaks = c(0, 1), labels = c("Low", "High"),
                        name = "Channel\nproportion",
                        guide = guide_colourbar(barwidth = 0.4, barheight = 4, ticks = FALSE)) +
  geom_vline(xintercept = 0, colour = GRID, linewidth = 0.3) +
  labs(title = "SHAP feature importance",
       x = "SHAP value (probability units)", y = NULL, caption = caption) +
  theme_j()
ggsave(file.path(output_path, "model_SHAP_plot.pdf"), p,
       width = 130, height = 90, units = "mm")
