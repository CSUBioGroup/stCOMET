# stCOMET：Co-expression Module-Enhanced Multi-view Contrastive Learning for Spatial Domain Identification in Spatial Transcriptomics
# Overview
Spatial transcriptomics measures gene expression while preserving tissue coordinates, and spatial domain identification—partitioning tissue into regions with coherent transcriptional programs—has become a foundational task for downstream analyses such as spatially resolved differential expression and cell–cell communication inference. However, current methods face two key limitations: at the feature level, highly variable gene sets are used without co-expression-based refinement, allowing noisy or isolated genes to degrade learned representations; at the representation level, embeddings are typically learned from a single spatial graph view, offering no explicit guarantee of stability under graph perturbation. To address these gaps, we present stCOMET, a spatial representation learning framework that couples co-expression-guided gene selection—which retains only genes belonging to coherent co-expression modules—with graph-based multi-view augmentation and a neighbourhood-aware contrastive objective that enforces representation consistency across complementary spatial views. Across 12 human dorsolateral prefrontal cortex sections and five MERFISH hypothalamic sections, stCOMET achieved the best average performance among eight evaluated methods, as measured by adjusted Rand index, normalized mutual information, completeness and homogeneity. stCOMET further identified a periventricular-region-associated MERFISH domain whose marker genes were enriched for neuropeptide signalling and hormone secretion, demonstrating the biological interpretability of the inferred spatial organization. These results establish stCOMET as a robust and interpretable framework for spatial domain identification across sequencing- and imaging-based spatial transcriptomics platforms. The implementation code is anonymously available at https://anonymous.4open.science/r/stCOMET
<p align="center">
  <img src="figures/stCOMET_fig.png" width="900">
</p>

# Requirements
Python >= 3.10

Core dependencies:
anndata==0.11.4
scanpy==1.11.5
squidpy==1.6.5
numpy==1.26.4
pandas==2.3.3
scipy==1.15.3
scikit-learn==1.7.2
scikit-image==0.25.2
scikit-misc==0.5.2
matplotlib==3.10.8
seaborn==0.13.2
networkx==3.4.2
igraph
leidenalg
umap-learn==0.5.11
h5py==3.16.0
tqdm==4.67.3

Deep learning dependencies:
torch==2.5.0+cu124
torchvision==0.20.0+cu124
torchaudio==2.5.0+cu124
torch-geometric==2.7.0
torch_scatter==2.1.2+pt25cu124
torch_sparse==0.6.18+pt25cu124
faiss-gpu==1.7.2

Spatial transcriptomics / image-related dependencies:
spatialdata==0.5.0
spatial_image==1.2.3
multiscale_spatial_image==2.0.3
ome-zarr==0.11.1
zarr==2.18.3
numcodecs==0.13.1
dask==2024.11.2
dask-image==2025.11.0
xarray==2025.6.1
tifffile==2025.5.10
pillow==12.1.1

Optional dependencies:
rpy2
POT==0.9.6.post1
geopandas==1.1.3
shapely==2.1.2
pyproj==3.7.1
