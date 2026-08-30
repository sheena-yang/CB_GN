################################################################################
# Fig 1 & Fig S1 | snRNA-seq of human cerebellum: clustering, annotation and plots
################################################################################

library(Seurat)
library(ggplot2)
library(dplyr)
library(harmony)
library(scDblFinder)
library(BiocParallel)
library(SingleR)
library(SingleCellExperiment)

set.seed(123)

input_path  <- "data"
output_path <- "results"
dir.create(output_path, showWarnings = FALSE, recursive = TRUE)

################################################################################
# 1. Quality control
################################################################################

sce_n <- readRDS(file.path(input_path, "cerebellum_snRNA_raw.RDS"))

sce_n[["percent.mt"]] <- PercentageFeatureSet(sce_n, pattern = "^MT-")
sce_n <- subset(sce_n, subset = nCount_RNA > 300 & nCount_RNA < 30000 &
                                nFeature_RNA > 200 & nFeature_RNA < 7000 &
                                percent.mt < 10)

n_cells <- rowSums(sce_n@assays$RNA@counts > 0)
sce_n <- subset(sce_n, features = names(n_cells)[n_cells >= 3])

################################################################################
# 2. Doublet detection
################################################################################

sce <- as.SingleCellExperiment(sce_n)
sce <- scDblFinder(sce, samples = "Sample", BPPARAM = MulticoreParam(13))
sce <- as.Seurat(sce)
sce <- subset(sce, subset = scDblFinder.class == "singlet")

################################################################################
# 3. Normalization, dimensionality reduction and clustering
################################################################################

DefaultAssay(sce) <- "RNA"
sce <- NormalizeData(sce, normalization.method = "LogNormalize", scale.factor = 10000)
sce <- FindVariableFeatures(sce, selection.method = "vst", nfeatures = 2000)
sce <- ScaleData(sce)
sce <- RunPCA(sce, features = VariableFeatures(sce), npcs = 50)
sce <- RunTSNE(sce, dims = 1:20)
sce <- RunUMAP(sce, dims = 1:20)

sce <- RunHarmony(sce, c("Sample"))
sce <- RunTSNE(sce, reduction = "harmony", dims = 1:20)
sce <- RunUMAP(sce, reduction = "harmony", dims = 1:20)

sce <- FindNeighbors(sce, reduction = "harmony", dims = 1:15)
sce <- FindClusters(sce, resolution = 1,   graph.name = "RNA_snn")
sce <- FindClusters(sce, resolution = 0.5, graph.name = "RNA_snn")

################################################################################
# 4. Cluster markers and reference-based annotation
################################################################################

markers <- FindAllMarkers(sce, only.pos = TRUE, test.use = "wilcox",
                          logfc.threshold = 0.25, return.thresh = 0.01)

ref_nn        <- readRDS(file.path(input_path, "Aldinger_CB_snRNA_reference.rds"))
ref_sce       <- SingleCellExperiment(list(logcounts = GetAssayData(ref_nn, slot = "data")))
ref_sce$label <- ref_nn@meta.data$fig_cell_type

pred <- SingleR(test = GetAssayData(sce, slot = "data"), ref = ref_sce, labels = ref_sce$label,
                BPPARAM = MulticoreParam(workers = 8), fine.tune = FALSE)
sce$SingleR_nn <- pred$labels[match(colnames(sce), rownames(pred))]

################################################################################
# 5. Cluster annotation
################################################################################

res <- as.integer(as.character(sce$RNA_snn_res.0.5))
sce$cluster <- case_when(
  res %in% c(0, 1, 2, 3, 4, 5, 7, 8) ~ "Granule",
  res == 6                           ~ "Granule-diff",
  res %in% c(9, 10)                  ~ "others"
)

################################################################################
# 6. Figures
################################################################################

## UMAP of annotated clusters
p_umap <- DimPlot(sce, reduction = "umap", group.by = "cluster", raster = FALSE)
ggsave(file.path(output_path, "celltype_umap.pdf"), p_umap, width = 5, height = 5)

## Marker dot plot
marker_genes <- c("GRIK2","RBFOX1","GABRA6","CALB1","PCP4","SORCS3","PTPRK","NXPH1","SNTG1",
                  "PLP1","MOG","PDGFRA","GFAP","AQP4","VIT","LGR6","CSF1R","C3","CLDN5","PDGFRB")
marker_types <- rep(c("Granule","Purkinje","MLI","OL","OPC","AS","BG","Mic","Vascular"),
                    times = c(3, 2, 4, 2, 1, 2, 2, 2, 2))

p_dot <- DotPlot(sce, features = split(marker_genes, marker_types),
                 group.by = "cluster", dot.min = 0) +
  theme(strip.text.x  = element_text(size = 10),
        axis.text.x   = element_text(color = "black", size = 10, angle = 90, vjust = 0.5, hjust = 1),
        panel.border  = element_rect(color = "black"),
        panel.spacing = unit(1, "mm"),
        axis.title    = element_blank())
ggsave(file.path(output_path, "celltype_marker.pdf"), p_dot, width = 7.9, height = 2)

## Cluster composition per sample
prop_sample <- as.data.frame(table(sce$Sample, sce$cluster)) %>%
  setNames(c("Sample", "Cluster", "Count")) %>%
  group_by(Sample) %>%
  mutate(Proportion = Count / sum(Count)) %>%
  ungroup()

p_sample <- ggplot(prop_sample, aes(x = Sample, y = Proportion, fill = Cluster)) +
  geom_col(width = 0.7) +
  theme_classic(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Sample", y = "Proportion", fill = "Cluster")
ggsave(file.path(output_path, "barplot_celltype_sample.pdf"), p_sample, width = 6, height = 4)

## Cluster composition by age group
prop_group <- as.data.frame(table(sce$Group, sce$cluster)) %>%
  setNames(c("Group", "Cluster", "Count")) %>%
  group_by(Group) %>%
  mutate(Proportion = Count / sum(Count)) %>%
  ungroup()

p_group <- ggplot(prop_group, aes(x = Group, y = Proportion, fill = Cluster)) +
  geom_col(width = 0.7) +
  theme_classic(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Age group", y = "Proportion", fill = "Cluster")
ggsave(file.path(output_path, "barplot_celltype_group.pdf"), p_group, width = 4, height = 4.5)
