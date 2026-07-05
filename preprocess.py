import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
import pandas as pd

def permute_stcomet_features(feature): 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated 

def construct_stcomet_spatial_graph(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    adata.obsm['adj'] = adj
    
def construct_stcomet_knn_graph(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    adata.obsm['adj'] = adj
    print('Graph constructed!')   

def _to_stcomet_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _smooth_stcomet_expression_spatially(adata, n_neighbors=6, alpha=0.5):
    from sklearn.neighbors import NearestNeighbors
    X = _to_dense(adata.X).astype(np.float32)
    spatial = adata.obsm["spatial"]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(spatial)
    _, idx = nbrs.kneighbors(spatial)
    neigh_idx = idx[:, 1:]
    X_neigh = X[neigh_idx].mean(axis=1)
    X_smooth = alpha * X + (1 - alpha) * X_neigh
    return X_smooth


def _filter_stcomet_genes_by_coexpression_modules(
    X,
    gene_names,  
    min_module_size=30,
    min_keep_genes=500,
    leiden_resolution=1.0,
    gene_neighbors=15,
    n_pcs=30,
):
    
    import scanpy as sc
    import pandas as pd
    import numpy as np
    X = np.asarray(X, dtype=np.float32)
    G = X.T
    G = G - G.mean(axis=1, keepdims=True)
    std = G.std(axis=1, keepdims=True)
    std[std == 0] = 1
    G = G / std
    adata_gene = sc.AnnData(G)
    adata_gene.obs_names = np.asarray(gene_names).astype(str)
    n_pcs_use = min(n_pcs, G.shape[0] - 1, G.shape[1] - 1)
    if n_pcs_use >= 2:
        sc.pp.pca(adata_gene, n_comps=n_pcs_use)
        sc.pp.neighbors(
            adata_gene,
            n_neighbors=gene_neighbors,
            n_pcs=n_pcs_use,
            use_rep="X_pca",
        )
    else:
        sc.pp.neighbors(
            adata_gene,
            n_neighbors=gene_neighbors,
            use_rep="X",
        )
    sc.tl.leiden(
        adata_gene,
        resolution=leiden_resolution,
        key_added="gene_module",
        random_state=0,
    )
    labels = adata_gene.obs["gene_module"].astype(int).values
    module_sizes = np.bincount(labels)
    keep_modules = np.where(module_sizes >= min_module_size)[0]
    keep_mask = np.isin(labels, keep_modules)
    keep_genes = np.asarray(gene_names)[keep_mask]
    if len(keep_genes) < min_keep_genes:
        print(
            f"[GeneModule-Leiden] only keep {len(keep_genes)} genes, "
            f"fallback to original HVGs."
        )
        keep_genes = np.asarray(gene_names)
        keep_mask = np.ones(len(gene_names), dtype=bool)
    print(
        f"[GeneModule-Leiden] modules={len(module_sizes)}, "
        f"large_modules={len(keep_modules)}, "
        f"selected_genes={len(keep_genes)}"
    )
    module_df = pd.DataFrame({
        "gene": np.asarray(gene_names),
        "module": labels,
        "module_size": module_sizes[labels],
        "selected": keep_mask,
    })
    return list(keep_genes), labels, module_sizes, module_df

def preprocess_stcomet(
    adata,
    use_spatial_smooth=True,
    smooth_neighbors=6,
    smooth_alpha=0.5,
    n_top_genes=3000,
    use_gene_module_filter=True,
    min_module_size=30,
    min_keep_genes=500,
    leiden_resolution=1.0,
    gene_neighbors=15,
    n_pcs=30,
):
    adata_for_hvg = adata.copy()
    if use_spatial_smooth:
        print("[Preprocess] use spatial smoothing before HVG.")
        adata_for_hvg.X = _smooth_stcomet_expression_spatially(
            adata_for_hvg,
            n_neighbors=smooth_neighbors,
            alpha=smooth_alpha,
        )
    else:
        print("[Preprocess] skip spatial smoothing.")

    sc.pp.highly_variable_genes(
        adata_for_hvg,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    hvg_genes = adata_for_hvg.var_names[adata_for_hvg.var["highly_variable"]].tolist()
    if use_gene_module_filter:
        tmp = adata_for_hvg[:, hvg_genes].copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        X_hvg = _to_stcomet_dense(tmp.X)
        keep_genes, module_labels, module_sizes, module_df = _filter_stcomet_genes_by_coexpression_modules(
            X_hvg,
            tmp.var_names,
            min_module_size=min_module_size,
            min_keep_genes=min_keep_genes,
            leiden_resolution=leiden_resolution,
            gene_neighbors=gene_neighbors,
            n_pcs=n_pcs,
        )
        adata.uns["gene_module_df"] = module_df
        adata.uns["gene_module_labels_hvg"] = {
            gene: int(label)
            for gene, label in zip(tmp.var_names, module_labels)
        }
        adata.uns["gene_module_sizes"] = module_sizes.tolist()

    else:
        print("[GeneModule] skip gene module filtering.")
        keep_genes = hvg_genes

   
    adata.var["highly_variable"] = False
    adata.var.loc[keep_genes, "highly_variable"] = True
    print(f"[Preprocess] final selected genes for Encoder: {len(keep_genes)}")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def filter_stcomet_genes_by_moranI(adata, percent=95):
    import squidpy as sq
    import numpy as np
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    if 'spatial_neighbors' not in adata_hvg.uns:
        sq.gr.spatial_neighbors(adata_hvg)
    sq.gr.spatial_autocorr(adata_hvg, mode='moran')
    moranI = adata_hvg.uns['moranI']['I']
    n_genes = len(moranI)
    k = int(n_genes * percent / 100)
    idx_sorted = np.argsort(moranI)
    idx_keep = idx_sorted[-k:]
    keep_genes = adata_hvg.var_names[idx_keep]
    print(f"[MoranI] percent={percent}  selected genes: {len(keep_genes)}")
    print(f"Top {percent}% MoranI  {k} genes")
    adata.var['highly_variable'] = False
    adata.var.loc[keep_genes, 'highly_variable'] = True
    return adata   

 
def get_stcomet_features(adata):
    adata_Vars =  adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    feat_a = permute_stcomet_features(feat)
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def normalize_stcomet_adjacency(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_stcomet_adjacency(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_stcomet_adjacency(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def _stcomet_sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_stcomet_sparse_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return _stcomet_sparse_mx_to_torch_sparse_tensor(adj_normalized)    

def fix_stcomet_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    try:
       torch.set_num_interop_threads(1)
    except RuntimeError:
       pass
    torch.use_deterministic_algorithms(True, warn_only=False)
    
