# @title Graph creation code, convert frame from molecular dynamics simulation into Graphs readable - Created by Fodil Azzaz, PhD

"""
Molecular Dynamics Graph Converter
Copyright (c) 2025 Fodil Azzaz - All Rights Reserved
Non-commercial use only

Converts MD simulation frames into EquiformerV2-compatible graphs
with 13D scalar features and non-covalent edge detection.

Original Methodology:
- 13-dimensional feature engineering for biomolecules
- Non-covalent edge detection with KD-tree optimization  
- Radial basis functions for edge features
- Google Colab-optimized pipeline
"""

# === IMPORTS ===
import os
import glob
import numpy as np
from scipy.spatial import cKDTree
import MDAnalysis as mda
import torch
from torch_geometric.data import Data
from google.colab import files
from google.colab import drive
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === CONSTANTS === The resnames depends on the force field used (here CHARMM36m)
LIPID_RESNAMES = ['POPC', 'CHL1', 'ANE5AC', 'CER160', 'BGLC', 'BGAL', 'BGALNA', 'POPE', 'POPS', 'CHOL']
WATER_RESNAMES = ['TIP3', 'SOL', 'WAT', 'HOH', 'TIP4P']
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
SEGIDS = ['PROA', 'PROB', 'GLIP', 'GLIZ', 'POPC', 'CHL1']  # CHANGED: PROD → PROB

# === FIXED PSF PARSING ===
def parse_psf_charges_masses(psf_filename):
    """Parse atom charges and masses from PSF file - FIXED VERSION"""
    charges, masses = {}, {}
    atoms_parsed = 0

    try:
        with open(psf_filename, 'r') as f:
            lines = f.readlines()

        in_atoms = False

        for line in lines:
            if '!NATOM' in line:
                in_atoms = True
                print("✅ Found ATOM section in PSF")
                continue
            elif in_atoms and ('!NBOND' in line or not line.strip()):
                in_atoms = False
                break

            if in_atoms and line.strip() and not line.startswith('!'):
                parts = line.split()

                # YOUR PSF FORMAT: [index, segid, resid, resname, atom_name, atom_type, charge, mass, ...]
                if len(parts) >= 8:
                    resname = parts[3]      # Column 3: resname (POPC)
                    atom_name = parts[4]    # Column 4: atom_name (C15)
                    charge_str = parts[6]   # Column 6: charge (-0.350000)
                    mass_str = parts[7]     # Column 7: mass (12.0107)

                    key = f"{resname}_{atom_name}"

                    try:
                        charge_val = float(charge_str)
                        mass_val = float(mass_str)

                        charges[key] = charge_val
                        masses[key] = mass_val
                        atoms_parsed += 1

                        # Print first few to verify it's working
                        if atoms_parsed <= 3:
                            print(f"   ✅ Sample: {key} → charge={charge_val:.3f}, mass={mass_val:.1f}")

                    except ValueError as e:
                        print(f"    Could not parse charge/mass for {key}: {charge_str}, {mass_str}")
                        continue

                else:
                    if atoms_parsed == 0:  # Only show this warning if we haven't parsed any atoms yet
                        print(f"    Skipping line with only {len(parts)} columns")

        print(f"✅ PSF parsed: {atoms_parsed} atom charges/masses")

        # VERIFY we got charges
        if atoms_parsed > 0:
            sample_charges = list(charges.values())[:5]
            charge_min = min(sample_charges)
            charge_max = max(sample_charges)
            print(f"   📊 Charge range: {charge_min:.3f} to {charge_max:.3f}")

            if charge_min == 0.0 and charge_max == 0.0:
                print("❌ WARNING: All charges are zero! PSF parsing may have failed.")
        else:
            print("❌ NO CHARGES PARSED! Using fallback...")
            charges, masses = create_fallback_charges_masses()

    except Exception as e:
        print(f"❌ PSF parsing error: {e}")
        charges, masses = create_fallback_charges_masses()

    return charges, masses

def create_fallback_charges_masses():
    """Create reasonable fallback charges when PSF parsing fails"""
    print("🔄 Creating fallback charges based on atom types...")

    charges, masses = {}, {}

    # Common atomic charges (approximate) - more realistic values
    atom_charges = {
        # Backbone atoms
        'N': -0.47, 'CA': 0.07, 'C': 0.51, 'O': -0.51, 'OXT': -0.51,
        # Hydrogens
        'H': 0.31, 'HA': 0.09, 'HB': 0.09, 'HG': 0.09, 'HD': 0.09, 'HE': 0.09,
        # Carbons
        'CB': 0.05, 'CG': -0.08, 'CD': -0.18, 'CE': -0.30, 'CZ': 0.25,
        # Oxygens
        'OG': -0.66, 'OG1': -0.66, 'OD1': -0.76, 'OD2': -0.76, 'OE1': -0.76, 'OE2': -0.76,
        # Nitrogens
        'ND1': -0.40, 'ND2': -0.60, 'NE': -0.70, 'NE1': -0.70, 'NE2': -0.70, 'NZ': -0.80,
        # Sulfurs
        'SG': -0.16, 'SD': 0.45,
        # Phosphorus (lipids)
        'P': 1.50,
        # Common lipid atoms
        'C': 0.00, 'O': -0.50, 'N': -0.30, 'H': 0.30
    }

    # Common masses
    atom_masses = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.06, 'P': 30.974
    }

    # Create entries for common atom types across all residues
    all_residues = AMINO_ACIDS + LIPID_RESNAMES + WATER_RESNAMES

    for atom_type, charge in atom_charges.items():
        for resname in all_residues:
            key = f"{resname}_{atom_type}"
            charges[key] = charge

            # Guess mass from atom type
            if atom_type.startswith('H'):
                masses[key] = atom_masses['H']
            elif atom_type.startswith('C'):
                masses[key] = atom_masses['C']
            elif atom_type.startswith('N'):
                masses[key] = atom_masses['N']
            elif atom_type.startswith('O'):
                masses[key] = atom_masses['O']
            elif atom_type.startswith('S'):
                masses[key] = atom_masses['S']
            elif atom_type.startswith('P'):
                masses[key] = atom_masses['P']
            else:
                masses[key] = 12.011  # Default carbon mass

    print(f"✅ Created fallback charges for {len(charges)} atom types")
    return charges, masses

def parse_psf_bonds(psf_filename):
    """Parse REAL covalent bonds from PSF"""
    bonds = []
    try:
        with open(psf_filename, 'r') as f:
            lines = f.readlines()

        in_bonds = False
        for line in lines:
            if '!NBOND' in line:
                in_bonds = True
                continue
            if in_bonds:
                if not line.strip() or '!NTHETA' in line:
                    break
                if line.strip().startswith('!'):
                    continue
                parts = line.split()
                for i in range(0, len(parts), 2):
                    if i+1 < len(parts):
                        try:
                            a1 = int(parts[i]) - 1  # Convert to 0-based indexing
                            a2 = int(parts[i+1]) - 1
                            bonds.append((a1, a2))
                        except ValueError:
                            continue
        print(f"✅ {len(bonds)} covalent bonds parsed from PSF")
    except Exception as e:
        print(f"❌ Bond parsing failed: {e}")
    return bonds

# === EQUIFORMERV2 FEATURE ENGINEERING ===
def get_element_from_atom_name(atom_name, resname):
    """Smart element detection from atom names"""
    clean_name = ''.join([c for c in atom_name if not c.isdigit()]).strip()
    element_map = {
        'N': 'N', 'CA': 'C', 'C': 'C', 'O': 'O', 'OXT': 'O',
        'CB': 'C', 'CG': 'C', 'CD': 'C', 'CE': 'C', 'CZ': 'C',
        'CG1': 'C', 'CG2': 'C', 'CD1': 'C', 'CD2': 'C',
        'OG': 'O', 'OG1': 'O', 'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O',
        'SD': 'S', 'SG': 'S', 'H': 'H', 'HA': 'H', 'HB': 'H', 'HG': 'H', 'HD': 'H',
        'P': 'P', 'OW': 'O', 'HW1': 'H', 'HW2': 'H'
    }
    if clean_name in element_map:
        return element_map[clean_name]
    if clean_name.startswith('H'): return 'H'
    elif clean_name.startswith('C'): return 'C'
    elif clean_name.startswith('N'): return 'N'
    elif clean_name.startswith('O'): return 'O'
    elif clean_name.startswith('S'): return 'S'
    elif clean_name.startswith('P'): return 'P'
    return 'C'

def get_atomic_number_from_element(element):
    """Convert element symbol to atomic number"""
    element_to_number = {
        'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'S': 16.0, 'P': 15.0
    }
    return element_to_number.get(element, 6.0)

def get_residue_hydrophobicity(resname):
    """Kyte-Doolittle hydrophobicity scale - Normalized to [-1, 1]"""
    hydrophobicity_scale = {
        'ALA': 1.8, 'VAL': 4.2, 'LEU': 3.8, 'ILE': 4.5, 'PHE': 2.8, 'TRP': -0.9,
        'MET': 1.9, 'PRO': -1.6, 'GLY': -0.4, 'SER': -0.8, 'THR': -0.7, 'CYS': 2.5,
        'TYR': -1.3, 'ASN': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'GLU': -3.5,
        'LYS': -3.9, 'ARG': -4.5, 'HIS': -3.2
    }
    raw_value = hydrophobicity_scale.get(resname, 0.0)
    return max(-1.0, min(1.0, raw_value / 4.5))

def get_residue_charge_category(resname):
    """Categorize residues by charge"""
    positive = ['LYS', 'ARG', 'HIS', 'HSD', 'HSE', 'HSP']
    negative = ['ASP', 'GLU']
    if resname in positive: return 1.0
    if resname in negative: return -1.0
    return 0.0

def create_equiformerv2_features(atom, psf_charges, psf_masses):
    """13-dimensional features for EquiformerV2 generative modeling"""
    # Create the SAME key format as used in PSF parsing
    key = f"{atom.resname}_{atom.name}"

    # 1. Physical properties from PSF
    charge = psf_charges.get(key, 0.0)
    mass = psf_masses.get(key, 12.0) / 100.0  # Normalized

    # 2. Element information
    element = get_element_from_atom_name(atom.name, atom.resname)
    atomic_number = get_atomic_number_from_element(element) / 10.0

    # 3. Residue properties
    residue_hydrophobicity = get_residue_hydrophobicity(atom.resname)
    residue_charge_category = get_residue_charge_category(atom.resname)

    # 4. Structural features
    is_backbone = 1.0 if atom.name in ['N', 'CA', 'C', 'O', 'OXT'] else 0.0
    is_sidechain = 1.0 if atom.name not in ['N', 'CA', 'C', 'O', 'OXT'] and atom.resname in AMINO_ACIDS else 0.0

    # 5. Segid one-hot encoding
    segid = getattr(atom, 'segid', 'other')
    segid_features = [1.0 if segid == seg else 0.0 for seg in SEGIDS]

    features = [
        charge, mass, atomic_number, residue_hydrophobicity,
        residue_charge_category, is_backbone, is_sidechain, *segid_features
    ]

    return features

def radial_basis_functions(distance, num_rbf=16, rbf_min=0.0, rbf_max=20.0):
    """Convert distance to radial basis functions for edges"""
    centers = np.linspace(rbf_min, rbf_max, num_rbf)
    width = (rbf_max - rbf_min) / num_rbf
    rbf_values = np.exp(-((distance - centers) ** 2) / (2 * width ** 2))
    return rbf_values.tolist()

# === GRAPH CREATION WITH NON-COVALENT EDGES ===

def create_non_covalent_edges(positions, segids, max_distance=6.0):
    """Create edges between atoms from different molecules with detailed tracking - FIXED"""
    print(f"     Creating NON-COVALENT edges (≤ {max_distance}Å)...")

    non_covalent_edges = []
    positions_np = np.array(positions)

    # This section depends on the "segid" contained in your system
    pair_counts = {
        'PROA_GLIZ': 0, 'GLIZ_PROA': 0, 
        'PROB_GLIZ': 0, 'GLIZ_PROB': 0,  
        'PROA_PROB': 0, 'PROB_PROA': 0,  
        'PROA_GLIP': 0, 'GLIP_PROA': 0,
        'PROB_GLIP': 0, 'GLIP_PROB': 0,  
        'GLIP_GLIZ': 0, 'GLIZ_GLIP': 0,
        'OTHER': 0
    }

    # Use KD-tree for efficient neighbor finding
    tree = cKDTree(positions_np)
    pairs = tree.query_pairs(max_distance)

    for i, j in pairs:
        segid_i, segid_j = segids[i], segids[j]

        # Only create edges between different molecules
        if segid_i != segid_j:
            # Create pair key for tracking
            pair_key = f"{segid_i}_{segid_j}"
            reverse_key = f"{segid_j}_{segid_i}"

            # Count this pair type
            if pair_key in pair_counts:
                pair_counts[pair_key] += 1
            elif reverse_key in pair_counts:
                pair_counts[reverse_key] += 1
            else:
                pair_counts['OTHER'] += 1

            # FIX: Only add ONE undirected edge per pair, not two
            # Store as (min, max) to avoid duplicates
            non_covalent_edges.append([min(i, j), max(i, j)])

    # Print detailed breakdown
    print(f"     NON-COVALENT EDGE BREAKDOWN:")
    total_non_covalent = len(non_covalent_edges)
    for pair_type, count in pair_counts.items():
        if count > 0:
            percentage = (count / total_non_covalent) * 100 if total_non_covalent > 0 else 0
            print(f"       {pair_type}: {count} edges ({percentage:.1f}%)")

    print(f"    ✅ {total_non_covalent} non-covalent edges total")
    return non_covalent_edges

def create_edges_with_rbf(positions, psf_bonds, atom_index_to_node_index, segids, cutoff=6.0):
    """Create BOTH covalent AND non-covalent edges"""
    print(f"    🔗 Creating edges for {len(positions)} atoms...")

    # 1. Covalent edges (existing code)
    covalent_edges = []
    covalent_pairs = set()

    for a1, a2 in psf_bonds:
        if a1 in atom_index_to_node_index and a2 in atom_index_to_node_index:
            n1 = atom_index_to_node_index[a1]
            n2 = atom_index_to_node_index[a2]
            if n1 != n2:
                # FIX: Store covalent edges as undirected pairs too
                covalent_edges.append([min(n1, n2), max(n1, n2)])
                covalent_pairs.add((min(n1, n2), max(n1, n2)))

    print(f"    ✅ {len(covalent_edges)} covalent bonds")

    # 2. NEW: Non-covalent edges
    non_covalent_edges = create_non_covalent_edges(positions, segids, max_distance=cutoff)

    # 3. Combine both edge types and make undirected for PyG
    all_edges_undirected = covalent_edges + non_covalent_edges

    # Convert to PyG format: [src, tgt] and [tgt, src] for each edge
    all_edges_pyg = []
    for src, tgt in all_edges_undirected:
        all_edges_pyg.append([src, tgt])
        all_edges_pyg.append([tgt, src])

    total_covalent = len(covalent_edges)
    total_non_covalent = len(non_covalent_edges)
    total_edges_pyg = len(all_edges_pyg) // 2  # Divide by 2 since we have both directions

    print(f"    📈 EDGE SUMMARY:")
    print(f"       Covalent: {total_covalent} edges")
    print(f"       Non-covalent: {total_non_covalent} edges")
    print(f"       TOTAL: {total_edges_pyg} edges")
    if total_edges_pyg > 0:
        print(f"       Non-covalent ratio: {total_non_covalent/total_edges_pyg*100:.1f}%")

    return all_edges_pyg, covalent_pairs

def create_edge_features(positions, edges, covalent_pairs):
    """Create RBF edge features with direction vectors"""
    edge_scalar = []
    edge_vector = []

    for src, tgt in edges:
        diff = positions[tgt] - positions[src]
        dist = np.linalg.norm(diff)

        # Scalar: RBF + bond type
        rbf = radial_basis_functions(dist)
        is_covalent = 1.0 if (min(src, tgt), max(src, tgt)) in covalent_pairs else 0.0
        edge_scalar.append([*rbf, is_covalent])

        # Vector: normalized direction
        if dist > 1e-8:
            edge_vector.append(diff / dist)
        else:
            edge_vector.append([1.0, 0.0, 0.0])

    return edge_scalar, edge_vector

def create_equiformerv2_graph_from_frame(frame_idx, universe, psf_charges, psf_masses, psf_bonds,
                                       protein_selection='protein', environment_selection=None):
    """Create EquiformerV2 compatible graph with selection"""
    print(f"   Processing frame {frame_idx} for EquiformerV2...")
    universe.trajectory[frame_idx]

    # === ATOM SELECTION ===
    protein_atoms = universe.select_atoms(protein_selection)

    if environment_selection:
        environment_atoms = universe.select_atoms(environment_selection)
    else:
        environment_atoms = universe.select_atoms('not protein')

    # Combine all atoms
    all_atoms = protein_atoms + environment_atoms

    # Validate selection
    if len(all_atoms) == 0:
        print(f"❌ No atoms found for selection!")
        return None

    print(f"    • Total atoms: {len(all_atoms)}")
    print(f"    • Protein atoms: {len(protein_atoms)}")
    print(f"    • Environment atoms: {len(environment_atoms)}")
    print(f"    • Selection: {protein_selection}")
    if environment_selection:
        print(f"    • Environment: {environment_selection}")

    # === CREATE ATOM INDEX MAPS ===
    all_atom_indices = set(atom.index for atom in all_atoms)

    # Create mapping from atom index to atom object
    atom_index_to_atom = {}
    for atom in all_atoms:
        atom_index_to_atom[atom.index] = atom

    # === SMART BOND FILTERING - ONLY BONDS BETWEEN SELECTED ATOMS ===
    valid_psf_bonds = []
    fully_inside_bonds = 0
    boundary_bonds = 0

    for a1, a2 in psf_bonds:
        a1_in_selection = a1 in all_atom_indices
        a2_in_selection = a2 in all_atom_indices

        if a1_in_selection and a2_in_selection:
            valid_psf_bonds.append((a1, a2))
            fully_inside_bonds += 1
        elif a1_in_selection or a2_in_selection:
            # Boundary bonds - include them but mark as boundary
            valid_psf_bonds.append((a1, a2))
            boundary_bonds += 1

    print(f"    🔗 Bonds: {fully_inside_bonds} inside, {boundary_bonds} boundary")

    # === COLLECT ATOM DATA ===
    atom_positions = []
    atom_features = []
    atom_index_to_node_index = {}
    all_atom_segids = []
    all_atom_resnames = []
    all_atom_residues = []

    # Track charge statistics for debugging
    charge_values = []

    for node_idx, atom in enumerate(all_atoms):
        atom_positions.append(atom.position.copy())
        features = create_equiformerv2_features(atom, psf_charges, psf_masses)
        atom_features.append(features)
        atom_index_to_node_index[atom.index] = node_idx
        all_atom_segids.append(getattr(atom, 'segid', 'UNK'))
        all_atom_resnames.append(atom.resname)
        all_atom_residues.append(atom.resid)

        # Track charge for debugging
        charge_values.append(features[0])

    print(f"     Processed {len(atom_features)} atoms")
    print(f"     Feature dimension: {len(atom_features[0])}")
    print(f"     SEGIDs: {set(all_atom_segids)}")
    print(f"     Residues: {set(all_atom_resnames)}")

    # DEBUG: Check charge distribution
    charge_min = min(charge_values)
    charge_max = max(charge_values)
    charge_mean = sum(charge_values) / len(charge_values)
    print(f"     Charges: {charge_min:.3f} to {charge_max:.3f} (mean: {charge_mean:.3f})")

    # Create edges (BOTH covalent AND non-covalent) 
    edge_index, covalent_pairs = create_edges_with_rbf(atom_positions, valid_psf_bonds, atom_index_to_node_index, all_atom_segids)

    # Create edge features 
    edge_scalar, edge_vector = create_edge_features(atom_positions, edge_index, covalent_pairs)

    # Convert to tensors
    scalar_tensor = torch.FloatTensor(np.array(atom_features)).to(device)
    pos_tensor = torch.FloatTensor(np.array(atom_positions)).to(device)
    edge_index_tensor = torch.LongTensor(edge_index).t().contiguous().to(device)
    edge_scalar_tensor = torch.FloatTensor(np.array(edge_scalar)).to(device)
    edge_vector_tensor = torch.FloatTensor(np.array(edge_vector)).to(device).unsqueeze(1)

    # Create graph - COMPATIBLE WITH BOTH GNN AND EQUIFORMERV2
    graph = Data(
        # === CORE FEATURES (GNN) ===
        x=scalar_tensor,                    # [N, 13] node features for classic GNN
        pos=pos_tensor,                     # [N, 3] RAW coordinates
        edge_index=edge_index_tensor,       # [2, E] edges

        # === EQUIFORMERV2 COMPATIBLE FEATURES ===
        edge_attr=(edge_scalar_tensor, edge_vector_tensor),  # RBF + vectors for EquiformerV2

        # === METADATA ===
        original_positions=pos_tensor.clone(),  # Keep for reference
        num_atoms=len(atom_features),
        num_edges=edge_index_tensor.shape[1] // 2,
        frame_idx=frame_idx,
        time_ps=universe.trajectory.time,
        segids=all_atom_segids,
        resnames=all_atom_resnames,
        residues=all_atom_residues,
        num_boundary_bonds=boundary_bonds,
    )

    print(f"     Graph created: {graph.num_atoms} atoms, {graph.num_edges} edges")
    print(f"     Node features: {graph.x.shape}, Edge features: {graph.edge_attr[0].shape}")

    return graph

# === FILE HANDLING ===
def setup_google_drive_files(psf_filename, dcd_filename):
    """Setup MD files from Google Drive"""
    print("\n📁 SETTING UP GOOGLE DRIVE FILES...")
    drive.mount('/content/drive')

    psf_basename = os.path.basename(psf_filename) if psf_filename else None
    dcd_basename = os.path.basename(dcd_filename) if dcd_filename else None

    # Find files in Google Drive
    psf_path, dcd_path = None, None
    for root, dirs, files in os.walk('/content/drive/MyDrive'):
        if psf_basename in files:
            psf_path = os.path.join(root, psf_basename)
        if dcd_basename in files:
            dcd_path = os.path.join(root, dcd_basename)

    if not psf_path or not dcd_path:
        print("❌ Files not found in Google Drive!")
        return None, None

    # Copy to working directory
    import shutil
    shutil.copy(psf_path, psf_basename)
    shutil.copy(dcd_path, dcd_basename)

    print(f"✅ Files copied: {psf_basename}, {dcd_basename}")
    return psf_basename, dcd_basename

def setup_md_files(psf_filename=None, dcd_filename=None):
    """Setup MD files with Google Drive or upload"""
    if psf_filename or dcd_filename:
        return setup_google_drive_files(psf_filename, dcd_filename)

    # Find existing files
    psf_files = glob.glob("*.psf")
    dcd_files = glob.glob("*.dcd")
    if psf_files and dcd_files:
        return psf_files[0], dcd_files[0]

    # Upload new files
    print("\n📤 UPLOAD MD FILES")
    uploaded = files.upload()
    psf_file = next((f for f in uploaded if f.endswith('.psf')), None)
    dcd_file = next((f for f in uploaded if f.endswith('.dcd')), None)

    if not psf_file or not dcd_file:
        print("❌ Missing PSF or DCD file!")
        return None, None

    return psf_file, dcd_file

# === MAIN PIPELINE ===
def run_equiformerv2_pipeline(psf_filename=None, dcd_filename=None,
                            protein_selection='protein', environment_selection=None,
                            num_frames=3, frame_step=1):
    """Main pipeline for EquiformerV2 generative graph creation"""
    print(" EQUIFORMERV2 GENERATIVE GRAPH PIPELINE")
    print(f" Protein: {protein_selection}")
    print(f" Environment: {environment_selection if environment_selection else 'all non-protein'}")

    # Setup files
    psf_file, dcd_file = setup_md_files(psf_filename, dcd_filename)
    if not psf_file or not dcd_file:
        return None

    # Parse PSF WITH FIXED PARSING
    print("\n🔧 PARSING PSF FILE (FIXED VERSION)...")
    psf_charges, psf_masses = parse_psf_charges_masses(psf_file)
    psf_bonds = parse_psf_bonds(psf_file)

    # Load universe
    u = mda.Universe(psf_file, dcd_file)
    print(f"✅ System loaded: {len(u.atoms)} atoms, {len(u.trajectory)} frames")

    # Process frames
    graphs = []
    frame_indices = list(range(0, min(num_frames * frame_step, len(u.trajectory)), frame_step))

    for i, frame_idx in enumerate(frame_indices):
        graph = create_equiformerv2_graph_from_frame(
            frame_idx, u, psf_charges, psf_masses, psf_bonds,
            protein_selection=protein_selection,
            environment_selection=environment_selection
        )
        if graph is not None:
            graphs.append(graph)
            print(f"    ✅ Frame {frame_idx} completed ({i+1}/{len(frame_indices)})")

    # Save graphs
    if graphs:
        filename = f"equiformerv2_graphs_{len(graphs)}frames_WITH_NONCOVALENT.pt"
        torch.save(graphs, filename)
        print(f" Saved {len(graphs)} graphs as {filename}")

        try:
            files.download(filename)
            print(f"✅ File downloaded: {filename}")
        except:
            print(f"📁 File saved locally: {filename}")

    return graphs

# === EXECUTION ===
if __name__ == "__main__":
    !pip install -q MDAnalysis torch-geometric scipy

    # Your specific files
    PSF_FILENAME = "ionized.psf"    #your psf file path (in this exemple it is ionized.psf)
    DCD_FILENAME = "syt2_movie.dcd" #your dcd file path (in this exemple it is syt2_movie.dcd)

    graphs = run_equiformerv2_pipeline(
        psf_filename=PSF_FILENAME,
        dcd_filename=DCD_FILENAME,
        protein_selection='(segid PROA and resid 1150:1280) or (segid PROB and resid 30:85)', #The protein(s) you want to target
        environment_selection='segid GLIZ',  # Your specific environment
        num_frames=1, #total frame you want to convert into graphs
        frame_step=1 #frames you want to skip
    )




