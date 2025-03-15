import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def normalize_coordinates(X):
    """
    Normalize X coordinates to [0, 1] range.
    """
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def calculate_radius(X_norm, percentile=15):
    """
    Calculate the radius based on the given percentile of distances from the mean.
    """
    return np.percentile(np.linalg.norm(X_norm - X_norm.mean(axis=0), axis=1), percentile)


def find_neighbors_within_radius(X_norm, radius, high_expressing):
    """
    Find neighbors within the given radius for each high expressing cell.
    """
    nn = NearestNeighbors(radius=radius)
    nn.fit(X_norm)
    distances_dict = nn.radius_neighbors(X_norm[high_expressing], return_distance=True)
    return distances_dict[0], distances_dict[1]  # Distances and indices


def calculate_gradients_for_high_expressing_cells(X_norm, gene, distances, indices, high_expressing):
    """
    Calculate gradients for high expressing cells based on their neighbors.
    """
    dx_local = np.zeros_like(gene)
    dy_local = np.zeros_like(gene)
    high_expr_indices = np.where(high_expressing)[0]
    
    for idx, i in enumerate(high_expr_indices):
        # Get neighbors within radius
        neighbor_idx = indices[idx]  # Use idx because indices are already for high expressing cells
        neighbor_distances = distances[idx]

        # Filter out zero distances to avoid division by zero
        nonzero_mask = neighbor_distances > 1e-10
        neighbor_idx = neighbor_idx[nonzero_mask]
        neighbor_distances = neighbor_distances[nonzero_mask]

        if len(neighbor_idx) > 0:
            # Calculate gradients using valid neighbors
            dx = X_norm[neighbor_idx, 0] - X_norm[i, 0]
            dy = X_norm[neighbor_idx, 1] - X_norm[i, 1]
            dgene = gene[neighbor_idx] - gene[i]
            
            # Avoid division by zero
            valid_dx = np.abs(dx) > 1e-10
            valid_dy = np.abs(dy) > 1e-10
            
            if np.any(valid_dx):
                dx_local[i] = np.mean(dgene[valid_dx] / dx[valid_dx])
            if np.any(valid_dy):
                dy_local[i] = np.mean(dgene[valid_dy] / dy[valid_dy])
    
    # Debug prints you can add:
    print(f"Number of high expressing cells: {np.sum(high_expressing)}")
    print(f"Average number of neighbors: {np.mean([len(idx) for idx in indices])}")
    print(f"Sample gradients: dx={dx_local[high_expr_indices][:5]}, dy={dy_local[high_expr_indices][:5]}")
    
    return dx_local, dy_local

def visualize_gradients(X_norm, gene, dx_local, dy_local, high_expressing, radius, gene_idx=None, scale_factor=0.01):
    """
    Visualize spatial gradients with quiver plots and gradient magnitudes.
    
    Parameters
    ----------
    X_norm : array-like, shape (n_samples, 2)
        Normalized spatial coordinates
    gene : array-like, shape (n_samples,)
        Gene expression values
    dx_local : array-like, shape (n_samples,)
        X-direction gradients
    dy_local : array-like, shape (n_samples,)
        Y-direction gradients
    high_expressing : array-like, shape (n_samples,)
        Boolean mask indicating high expressing cells
    radius : float
        Radius used for neighbor search
    gene_idx : int or None, default=None
        Gene index for plot title
    scale_factor : float, default=0.01
        Scale factor for gradient arrows (smaller value = shorter arrows)
    """
    # Get indices of high expressing cells
    high_expr_indices = np.where(high_expressing)[0]
    
    # Scale gradients for visualization
    dx_scaled = dx_local * scale_factor
    dy_scaled = dy_local * scale_factor
    
    # Create quiver plot
    plt.figure(figsize=(10, 8))
    
    # Plot quiver only for high expressing cells
    plt.quiver(X_norm[high_expressing, 0], X_norm[high_expressing, 1],
              dx_scaled[high_expressing], dy_scaled[high_expressing],
              angles='xy', scale_units='xy',
              scale=np.max(np.sqrt(dx_scaled[high_expressing]**2 + dy_scaled[high_expressing]**2))*0.2,
              width=0.005,
              headwidth=3,
              headlength=5,
              headaxislength=4.5,
              color='red', alpha=0.5)
    
    # Add scatter plot colored by expression for all cells
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=gene,
               cmap='viridis', alpha=0.5)
    
    # Highlight high expressing cells
    plt.scatter(X_norm[high_expressing, 0], X_norm[high_expressing, 1],
               facecolors='none', edgecolors='black', s=100,
               label='High expressing cells')
    
    # Visualize radius for a sample high expressing point
    if np.any(high_expressing):
        sample_idx = high_expr_indices[len(high_expr_indices)//2]
        circle = plt.Circle((X_norm[sample_idx,0], X_norm[sample_idx,1]), radius,
                           fill=False, linestyle='--', color='gray',
                           label='Sample radius')
        plt.gca().add_patch(circle)
    
    plt.colorbar(label='Gene Expression')
    title = 'Gene Expression Gradient for High Expressing Cells'
    if gene_idx is not None:
        title += f' (Gene {gene_idx})'
    plt.title(title)
    plt.xlabel('Normalized Spatial X')
    plt.ylabel('Normalized Spatial Y')
    plt.legend()
    plt.show()
    
    # Plot gradient magnitudes
    gradient_norms_local = np.sqrt(dx_local*2 + dy_local*2)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_norm[:, 0], X_norm[:, 1], c=gradient_norms_local,
               cmap='viridis', alpha=0.8)
    
    plt.colorbar(label='Gradient Magnitude')
    title = 'Gene Expression Gradient Magnitude (High Expressing Cells)'
    if gene_idx is not None:
        title += f' (Gene {gene_idx})'
    plt.title(title)
    plt.xlabel('Normalized Spatial X')
    plt.ylabel('Normalized Spatial Y')
    plt.show()

def calculate_gradients_for_gene_matrix(X_norm, gene_matrix, percentile=75, gene_idx=None):
    """
    Calculate gradients for multiple genes based on their neighbors.
    """
    n_samples, n_genes = gene_matrix.shape
    
    # If gene_idx is specified, only calculate for that gene
    if gene_idx is not None:
        if not isinstance(gene_idx, int) or gene_idx < 0 or gene_idx >= n_genes:
            raise ValueError(f"gene_idx must be an integer between 0 and {n_genes-1}")
        
        gene = gene_matrix[:, gene_idx]
        high_expressing = gene > np.percentile(gene, percentile)
        
        # Calculate radius once since it only depends on spatial coordinates
        radius = calculate_radius(X_norm)
        
        # Find neighbors for high expressing cells
        distances, indices = find_neighbors_within_radius(X_norm, radius, high_expressing)
        
        # Calculate gradients for the specified gene
        dx_local, dy_local = calculate_gradients_for_high_expressing_cells(X_norm, gene, distances, indices, high_expressing)
        
        return dx_local, dy_local
    
    # Otherwise calculate for all genes
    dx_matrix = np.zeros_like(gene_matrix)
    dy_matrix = np.zeros_like(gene_matrix)
    
    # Calculate radius once since it only depends on spatial coordinates
    radius = calculate_radius(X_norm)
    
    for idx in range(n_genes):
        gene = gene_matrix[:, idx]
        high_expressing = gene > np.percentile(gene, percentile)
        
        # Find neighbors for high expressing cells
        distances, indices = find_neighbors_within_radius(X_norm, radius, high_expressing)
        
        # Calculate gradients for current gene
        dx_local, dy_local = calculate_gradients_for_high_expressing_cells(X_norm, gene, distances, indices, high_expressing)
        
        # Store results
        dx_matrix[:, idx] = dx_local
        dy_matrix[:, idx] = dy_local
    
    return dx_matrix, dy_matrix