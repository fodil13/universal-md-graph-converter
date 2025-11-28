# Molecular Dynamics Graph Converter

Convert MD simulation frames to graphs compatible with both EquiformerV2 and classic GNN
# Universal MD Graph Converter

**The bridge between molecular dynamics and modern graph neural networks**


## Why Researchers Choose This Converter

| Feature | Our Converter | 
|---------|---------------|
| **Dual Architecture Support** | ✅ EquiformerV2 + Classic GNNs 
| **13D Feature Engineering** | ✅ Charge, mass, hydrophobicity, etc. 
| **Non-covalent Edges** | ✅ KD-tree optimized interfaces 
| **Google Colab Ready** | ✅ One-click running (You will still have to adjust specific parameters depending on your simulated system)

Node Features (13 Dimensions)
1. Atomic charge (from PSF or intelligent estimation)

2. Atomic mass (normalized)

3. Atomic number (element encoding)

4. Residue hydrophobicity (Kyte-Doolittle scale)

5. Residue charge category (positive/negative/neutral)

6. Backbone indicator

7. Sidechain indicator
8-13. Molecular segment encoding (PROA, PROB, GLIP, GLIZ, POPC, CHL1)

Edge Features

- Radial basis functions (16D distance encoding)
- Covalent bond identificaiton
- Non-covalent interactions (KD-tree optimized)
- Normalized direction vectors (for equivariant networks)

Research Applications
- Protein-protein interaction analysis from MD trajectories
- Ligand binding site prediction with graph neural networks
- Molecular interface studies with chemical feature encoding

License
Academic Use - © 2025 Fodil Azzaz

This software is freely available for academic and research purposes. For commercial licensing, please contact azzaz.fodil@gmail.com

Acknowledgments

Built with the following open-source tools:

MDAnalysis 
PyTorch Geometric
Google Colab 
