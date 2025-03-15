import numpy as np
import matplotlib.pyplot as plt

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