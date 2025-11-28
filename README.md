# universal-md-graph-converter
Convert MD simulation frames into universal graphs compatible with both EquiformerV2 and classic GNNs
# Universal MD Graph Converter


**The bridge between molecular dynamics and modern graph neural networks**

Convert MD simulation frames into **universal graphs** compatible with **both EquiformerV2 and classic GNNs**.

## Why Researchers Choose This Converter

| Feature | Our Converter | Typical Alternatives |
|---------|---------------|---------------------|
| **Dual Architecture Support** | ✅ EquiformerV2 + Classic GNNs 
| **13D Feature Engineering** | ✅ Charge, mass, hydrophobicity, etc. 
| **Non-covalent Edges** | ✅ KD-tree optimized interfaces 
| **Google Colab Ready** | ✅ One-click running 

## Key Features

### **Dual GNN Compatibility**
```python
# Same graphs work with BOTH architectures:
from graph_creation_fodilazzaz import run_equiformerv2_pipeline

graphs = run_equiformerv2_pipeline(...)

# Use with EquiformerV2 (equivariant)
from equiformer_v2 import EquiformerV2
model = EquiformerV2()

# Use with Classic GNNs (GCN, GAT, GraphSAGE)  
from torch_geometric.nn import GCNConv
model = GCNConv(13, 64)
