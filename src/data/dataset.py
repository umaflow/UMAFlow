import io
import random
import warnings
import torch
import webdataset as wds
from collections import defaultdict 

from pathlib import Path
from torch.utils.data import Dataset

from src.data.data_utils import TensorDict, collate_entity
from src.constants import WEBDATASET_SHARD_SIZE, WEBDATASET_VAL_SIZE


class ProcessedLigandPocketDataset(Dataset):
    VALID_EMBEDDING_TYPES = (
        # Graph-level: one vector per complex. `.pt` files contain a dict keyed
        # by 'l0' / 'l1_norm' / ... / 'combined'.
        'invariant_complex_embedding',
        'ligand_invariant_embedding',
        # Atom-level: per-atom features derived from `atom_embeddings` of shape
        # [num_pocket_atoms + num_ligand_atoms, (lmax+1)^2, sphere_channels].
        # Pocket atoms come first, then ligand atoms.
        'atom_ligand',
    )
    ATOM_EMBEDDING_TYPES = ('atom_ligand',)

    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False, embeddings_dir=None,
                 embedding_type='ligand_invariant_embedding',
                 embedding_key='combined'):

        self.ligand_transform = ligand_transform
        self.pocket_transform = pocket_transform
        self.catch_errors = catch_errors
        self.pt_path = pt_path

        self.data = torch.load(pt_path)

        if embedding_type not in self.VALID_EMBEDDING_TYPES:
            raise ValueError(
                f"Invalid embedding_type '{embedding_type}'. "
                f"Must be one of {self.VALID_EMBEDDING_TYPES}."
            )
        self.embedding_type = embedding_type
        self.embedding_key = embedding_key

        # `complex_id` (e.g. 'GCR_HUMAN_507_777_0/4udd_A_rec_..._docked_0') is
        # the relative path under the embeddings dir. When present we look the
        # .pt file up directly; otherwise we fall back to the legacy
        # name-derived stem with an rglob-built index, which is much slower to
        # initialise on large embedding sets.
        self.has_complex_id = 'complex_id' in self.data['ligands']
        if embeddings_dir is not None:
            self.embeddings_dir = Path(embeddings_dir)
        else:
            self.embeddings_dir = None

        # add number of nodes for convenience
        for entity in ['ligands', 'pockets']:
            self.data[entity]['size'] = torch.tensor([len(x) for x in self.data[entity]['x']])
            self.data[entity]['n_bonds'] = torch.tensor([len(x) for x in self.data[entity]['bond_one_hot']])

    def _resolve_embedding_path(self, ligand):
        """Locate the .pt for this ligand. Prefers the relative `complex_id`
        path (fast, deterministic); falls back to the legacy stem-based
        rglob index for older dataset .pt files without `complex_id`.
        """
        complex_id = ligand.get('complex_id') if isinstance(ligand, dict) else None
        if complex_id is None:
            raise ValueError(
                "Ligand is missing 'complex_id' key. Cannot resolve embedding path. "
                "Please regenerate your dataset with `complex_id` included, "
                "or switch to an older dataset version that includes them."
            )
        p = self.embeddings_dir / f'{complex_id}.pt'
        return p if p.exists() else None

    def _load_embedding(self, ligand):
        emb_path = self._resolve_embedding_path(ligand)
        if emb_path is None:
            raise FileNotFoundError(
                f"No embedding file found for ligand '{ligand['name']}' "
                f"under {self.embeddings_dir}"
            )
        emb = torch.load(emb_path, weights_only=False)
        v = self._load_raw_embedding(emb)
        return v

    def _load_raw_embedding(self, emb):
        """Return the requested embedding without centering.

        Graph-level types return a 1D tensor [d]. Atom-level types return a
        2D tensor [N_ligand_heavy, d]. The new embedding files (regenerated
        on 2026-04-28 with `complex_id` keying) place pocket heavy atoms
        first then ligand heavy atoms in DrugFlow's atom order, so a plain
        `raw[n_p : n_p + n_l]` slice already aligns row-for-row with
        `ligand['x']`. No coordinate matching needed.
        """
        if self.embedding_type in ('invariant_complex_embedding',
                                   'ligand_invariant_embedding'):
            sub = emb[self.embedding_type]
            return sub[self.embedding_key] if isinstance(sub, dict) else sub

        n_p = int(emb['num_pocket_atoms'])
        n_l = int(emb['num_ligand_atoms'])
        raw = emb['atom_embeddings'][n_p:n_p + n_l]   # [N_ligand, (lmax+1)^2, C]
        return self._extract_l_component(raw, self.embedding_key)

    @staticmethod
    def _extract_l_component(raw, key):
        """Pick / norm one L-component from a [N, (lmax+1)^2, C] SH tensor.

        'l0' returns the scalar slice as [N, C]. 'lK_norm' returns the Euclidean
        norm over the 2K+1 components as [N, C]. 'combined' concatenates l0 with
        lK_norm for K=1..lmax → [N, (lmax+1)*C].

        'l0_l1' returns a dict {'l0': [N, C], 'l1': [N, 3, C]}. Empirically
        verified on UMA-S: slots 1,2,3 of atom_embeddings are already ordered
        as Cartesian (x, y, z), not the SH real-basis (y, z, x) convention — no
        permutation applied. (Rotation test: raw identity wins with mean error
        6e-5 vs ~1.45 for any permutation.)
        """
        total = raw.shape[1]
        lmax = int(round(total ** 0.5)) - 1
        if (lmax + 1) ** 2 != total:
            raise ValueError(
                f"atom_embeddings SH dim {total} is not (lmax+1)^2 for any integer lmax"
            )
        if key == 'l0':
            return raw[:, 0, :]
        if key == 'l0_l1':
            if lmax < 1:
                raise ValueError(f"Requested 'l0_l1' but lmax={lmax}")
            return {'l0': raw[:, 0, :], 'l1': raw[:, 1:4, :]}
        if key == 'combined':
            parts = [raw[:, 0, :]]
            for L in range(1, lmax + 1):
                parts.append(raw[:, L * L:(L + 1) ** 2, :].norm(dim=1))
            return torch.cat(parts, dim=-1)
        if key.startswith('l') and key.endswith('_norm'):
            L = int(key[1:-len('_norm')])
            if L < 1 or L > lmax:
                raise ValueError(f"Requested '{key}' but lmax={lmax}")
            return raw[:, L * L:(L + 1) ** 2, :].norm(dim=1)
        raise ValueError(f"Unknown embedding_key '{key}'")

    def __len__(self):
        return len(self.data['ligands']['name'])

    def __getitem__(self, idx):
        data = {}
        data['ligand'] = {key: val[idx] for key, val in self.data['ligands'].items()}
        data['pocket'] = {key: val[idx] for key, val in self.data['pockets'].items()}

        if self.embeddings_dir is not None:
            data['embedding'] = self._load_embedding(data['ligand'])
            # Atom-level alignment needs per-atom correspondence. Defensive
            # guard: with the regenerated v2 embeddings DrugFlow's heavy
            # atom count should always match `num_ligand_atoms` row-for-row,
            # but if it doesn't (legacy .pt, corrupt file) skip the sample.
            if self.embedding_type in self.ATOM_EMBEDDING_TYPES:
                emb = data['embedding']
                n_rows = emb['l0'].shape[0] if isinstance(emb, dict) else emb.shape[0]
                if n_rows != len(data['ligand']['x']):
                    return self[random.randint(0, len(self) - 1)]
        try:
            if self.ligand_transform is not None:
                data['ligand'] = self.ligand_transform(data['ligand'])
            if self.pocket_transform is not None:
                data['pocket'] = self.pocket_transform(data['pocket'])
        except (RuntimeError, ValueError) as e:
            if self.catch_errors:
                warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                              f"Returning random item instead")
                # replace bad item with a random one
                rand_idx = random.randint(0, len(self) - 1)
                return self[rand_idx]
            else:
                raise e
        return data

    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):

        out = {}
        for entity in ['ligand', 'pocket']:
            batch = [x[entity] for x in batch_pairs]

            if entity == 'ligand' and ligand_transform is not None:
                max_size = max(x['size'].item() for x in batch)
                # TODO: might have to remove elements from batch if processing fails, warn user in that case
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        if 'embedding' in batch_pairs[0]:
            embs = [x['embedding'] for x in batch_pairs]
            # Graph-level: [d] per sample → stack to [B, d].
            # Atom-level tensor: [N_i, d] → concat to [sum_i N_i, d].
            # Atom-level dict (e.g. 'l0_l1'): concat each component along atom axis.
            # The consumer pairs rows with ligand/pocket masks for scatter.
            if isinstance(embs[0], dict):
                out['embedding'] = {
                    k: torch.cat([e[k] for e in embs], dim=0) for k in embs[0]
                }
            elif embs[0].dim() == 1:
                out['embedding'] = torch.stack(embs)
            else:
                out['embedding'] = torch.cat(embs, dim=0)

        return out


class ClusteredDataset(ProcessedLigandPocketDataset):
    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False, embeddings_dir=None,
                 embedding_type='invariant_complex_embedding',
                 embedding_key='combined'):
        super().__init__(pt_path, ligand_transform, pocket_transform, catch_errors,
                         embeddings_dir=embeddings_dir,
                         embedding_type=embedding_type,
                         embedding_key=embedding_key)
        self.clusters = list(self.data['clusters'].values())

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, cidx):
        cluster_inds = self.clusters[cidx]
        # idx = cluster_inds[random.randint(0, len(cluster_inds) - 1)]
        idx = random.choice(cluster_inds)
        return super().__getitem__(idx)

class DPODataset(ProcessedLigandPocketDataset):
    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False):
        self.ligand_transform = ligand_transform
        self.pocket_transform = pocket_transform
        self.catch_errors = catch_errors
        self.pt_path = pt_path

        self.data = torch.load(pt_path)

        if not 'pockets' in self.data:
            self.data['pockets'] = self.data['pockets_w']
        if not 'ligands' in self.data:
            self.data['ligands'] = self.data['ligands_w']

        if (
            len(self.data["ligands"]["name"])
            != len(self.data["ligands_l"]["name"])
            != len(self.data["pockets"]["name"])
        ):
            raise ValueError(
                "Error while importing DPO Dataset: Number of ligands winning, ligands losing and pockets must be the same"
            )

        # add number of nodes for convenience
        for entity in ['ligands', 'ligands_l', 'pockets']:
            self.data[entity]['size'] = torch.tensor([len(x) for x in self.data[entity]['x']])
            self.data[entity]['n_bonds'] = torch.tensor([len(x) for x in self.data[entity]['bond_one_hot']])

    def __len__(self):
        return len(self.data["ligands"]["name"])

    def __getitem__(self, idx):
        data = {}
        data['ligand'] = {key: val[idx] for key, val in self.data['ligands'].items()}
        data['ligand_l'] = {key: val[idx] for key, val in self.data['ligands_l'].items()}
        data['pocket'] = {key: val[idx] for key, val in self.data['pockets'].items()}
        try:
            if self.ligand_transform is not None:
                data['ligand'] = self.ligand_transform(data['ligand'])
                data['ligand_l'] = self.ligand_transform(data['ligand_l'])
            if self.pocket_transform is not None:
                data['pocket'] = self.pocket_transform(data['pocket'])
        except (RuntimeError, ValueError) as e:
            if self.catch_errors:
                warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                              f"Returning random item instead")
                # replace bad item with a random one
                rand_idx = random.randint(0, len(self) - 1)
                return self[rand_idx]
            else:
                raise e
        return data
    
    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):

        out = {}
        for entity in ['ligand', 'ligand_l', 'pocket']:
            batch = [x[entity] for x in batch_pairs]

            if entity in ['ligand', 'ligand_l'] and ligand_transform is not None:
                max_size = max(x['size'].item() for x in batch)
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        return out

##########################################
############### WebDatasets ##############
##########################################

class ProteinLigandWebDataset(wds.WebDataset):
    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):
        return ProcessedLigandPocketDataset.collate_fn(batch_pairs, ligand_transform)


def wds_decoder(key, value):
    return torch.load(io.BytesIO(value))


def preprocess_wds_item(data):
    out = {}
    for entity in ['ligand', 'pocket']:
        out[entity] = data['pt'][entity]
        for attr in ['size', 'n_bonds']:
            if torch.is_tensor(out[entity][attr]):
                assert len(out[entity][attr]) == 0
                out[entity][attr] = 0

    return out


def get_wds(data_path, stage, ligand_transform=None, pocket_transform=None):
    current_data_dir = Path(data_path, stage)
    shards = sorted(current_data_dir.glob('shard-?????.tar'), key=lambda s: int(s.name.split('-')[-1].split('.')[0]))
    min_shard = min(shards).name.split('-')[-1].split('.')[0]
    max_shard = max(shards).name.split('-')[-1].split('.')[0]
    total_size = (int(max_shard) - int(min_shard) + 1) * WEBDATASET_SHARD_SIZE if stage == 'train' else WEBDATASET_VAL_SIZE

    url = f'{data_path}/{stage}/shard-{{{min_shard}..{max_shard}}}.tar'
    ligand_transform_wrapper = lambda _data: _data
    pocket_transform_wrapper = lambda _data: _data

    if ligand_transform is not None:
        def ligand_transform_wrapper(_data):
            _data['pt']['ligand'] = ligand_transform(_data['pt']['ligand'])
            return _data
        
    if pocket_transform is not None:
        def pocket_transform_wrapper(_data):
            _data['pt']['pocket'] = pocket_transform(_data['pt']['pocket'])
            return _data

    return (
        ProteinLigandWebDataset(url, nodesplitter=wds.split_by_node)
        .decode(wds_decoder)
        .map(ligand_transform_wrapper)
        .map(pocket_transform_wrapper)
        .map(preprocess_wds_item)
        .with_length(total_size)
    )
