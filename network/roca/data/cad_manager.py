import json
import pickle
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import torch

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import join_meshes_as_batch, Meshes
from pytorch3d.utils import ico_sphere


class CADManager:

    class NoCADError(KeyError):
        pass

    def __init__(
        self,
        db_file: Optional[str] = None,
        scene_file: Optional[str] = None,
        point_file: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        syms: Optional[Dict[str, Any]] = None,
        points: Optional[Dict[str, Any]] = None,
        grid_file: Optional[str] = None
    ):
        assert db_file is not None or data is not None, \
            'Provide db_file or data!'
        if db_file is not None:
            with open(db_file, 'rb') as f:
                models = pickle.load(f)
            self._data = defaultdict(lambda: {})
            self._syms = defaultdict(lambda: {})
            for model in models:
                key = (model['catid_cad'], model['id_cad'])
                mesh = Meshes(
                    [torch.from_numpy(model['verts'])],
                    [torch.from_numpy(model['faces'])]
                )
                self._data[model['category_id']][key] = mesh
                self._syms[model['category_id']][key] = model['sym']
            self._data = dict(self._data)
            self._syms = dict(self._syms)
        else:
            self._data = data
            self._syms = syms
            self._points = points

        self.dummy_model = ico_sphere()
        self.num_models = sum(len(v) for v in self._data.values())
        self.unique_cads = tuple(sorted(set(
            m for m in chain(*self._data.values())
        )))
        self._unique_meshes = None

        self.scene_alignments = None
        if scene_file is not None:
            with open(scene_file) as f:
                self.scene_alignments = json.load(f)

        if point_file is not None:
            self._points = defaultdict(lambda: {})
            with open(point_file, 'rb') as f:
                points = pickle.load(f)
                for p in points:
                    key = (p['catid_cad'], p['id_cad'])
                    try:
                        self.model_by_id(*key)
                    except CADManager.NoCADError:
                        print('skipping vacant point sample {}...'.format(key))
                        continue
                    cid = p['category_id']
                    self._points[cid][key] = torch.as_tensor(
                        p['points'], dtype=torch.float
                    )
            self._points = dict(self._points)

        if grid_file is not None:
            catid_by_category = {
                c: {vk[0] for vk in v.keys()}
                for c, v in self._data.items()
            }
            category_by_catid = {}
            for c, cats in catid_by_category.items():
                for cat in cats:
                    category_by_catid[cat] = c

            with open(grid_file, 'rb') as f:
                grid_data = pickle.load(f)

            self._grids = defaultdict(lambda: {})
            for (catid, id), grid in grid_data.items():
                try:
                    self.model_by_id(catid, id)
                except CADManager.NoCADError:
                    print('Skipping vacant volume sample {}...'.format(
                        (catid, id)
                    ))
                self._grids[category_by_catid[catid]][(catid, id)] = grid
            self._grids = dict(self._grids)

    def model_by_id(
        self,
        catid: str,
        id: str,
        category_id: int = None,
        use_dummy: bool = False,
        verbose: bool = False
    ) -> Meshes:
        key = (catid, id)
        if category_id is None:
            for d in self._data.values():
                if key in d:
                    # FIXME: this is hacky!
                    result = d[key]
                    if isinstance(result, list):
                        return result[0]
                    return result
            if use_dummy:
                if verbose:
                    print('Dummy used for {}'.format(key))
                return self.dummy_model
            else:
                raise CADManager.NoCADError('Model {} not found'.format(key))
        else:
            if use_dummy:
                return self._data[category_id].get(key, self.dummy_model)
            else:
                return self._data[category_id][key]

    def models_by_category(
        self,
        category_id: int
    ) -> Dict[Tuple[str, str], Meshes]:
        return self._data[category_id]

    def models_by_ids(
        self,
        catid_id_tuples: List[Tuple[str, str]],
        use_dummy: bool = False
    ) -> List[Meshes]:
        models = []
        for cat_id, id in catid_id_tuples:
            models.append(self.model_by_id(cat_id, id, use_dummy=use_dummy))
        return models

    def unique_meshes(self, reset_cache=False) -> List[Meshes]:
        use_cache = not reset_cache
        if not use_cache or self._unique_meshes is None:
            self._unique_meshes = self.models_by_ids(self.unique_cads)
        return self._unique_meshes

    # TODO: add symmetries and caching
    def batched_points_and_ids(
        self, volumes=False
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, str]]]:
        points = {}
        ids = {}
        all_points = self._grids if volumes else self._points
        collate_fn = (lambda x: x) if volumes else torch.stack
        for cat, items in all_points.items():
            points_cat = []
            ids_cat = []
            for k in sorted(items.keys()):
                points_cat.append(items[k])
                ids_cat.append(k)
            points[cat] = collate_fn(points_cat)
            ids[cat] = ids_cat
        return points, ids

    def batched_meshes_and_points(
        self,
        num_points: int,
        reset_cache: bool = False,
        ret_syms: bool = False
    ) -> Tuple:

        # Point sampling logic
        def create_batched_points(batched_meshes):
            batched_points = {}
            for category, meshes in batched_meshes.items():
                batched_points[category] = sample_points_from_meshes(
                    meshes, num_points
                )
            return batched_points

        # Handle the cached data
        from_cache = False
        if not reset_cache and hasattr(self, '_batched_cache'):
            batched_points = self._batched_cache[1]
            if next(iter(batched_points.values())).size(1) == num_points:
                from_cache = True

        # Create mesh and point batches from scratch
        if not from_cache:
            batched_meshes = {}
            batched_ids = {}
            for category, cad_dict in self._data.items():
                meshes = []
                ids = []
                for k in sorted(cad_dict.keys()):
                    meshes.append(cad_dict[k])
                    ids.append(k)
                batched_meshes[category] = join_meshes_as_batch(meshes)
                batched_ids[category] = ids
            batched_points = create_batched_points(batched_meshes)
            self._batched_cache = (batched_meshes, batched_points, batched_ids)

        # Handle symmetry
        if ret_syms:
            if reset_cache or not hasattr(self, '_sym_cache'):
                batched_syms = {}
                assert self._syms is not None, 'Symmetries are not registered!'
                for category, sym_dict in self._syms.items():
                    syms = []
                    for k in sorted(sym_dict.keys()):
                        syms.append(sym_dict[k])
                    batched_syms[category] = syms
                self._sym_cache = batched_syms
            return (*self._batched_cache, self._sym_cache)

        # Return the cached data
        return self._batched_cache

    @classmethod
    def merge(cls, *cad_managers) -> 'CADManager':
        data = {}
        syms = {}
        points = {}

        categories = set()
        for cad_manager in cad_managers:
            categories.update(cad_manager._data.keys())

        for category in categories:
            datum = defaultdict(lambda: [])
            sym = defaultdict(lambda: [])
            point = defaultdict(lambda: [])
            for cad_manager in cad_managers:
                for k, v in cad_manager._data.get(category, {}).items():
                    datum[k].extend(v)
                    if cad_manager._points is not None:
                        point[k].extend(cad_manager._points[category][k])
                    if cad_manager._syms is not None:
                        sym[k] = cad_manager._syms[category][k]
            data[category] = dict(datum)
            if len(sym) > 0:
                syms[category] = dict(sym)
            if len(point) > 0:
                points[category] = dict(points)

        return cls(
            data=data,
            syms=syms if len(syms) > 0 else None,
            points=points if len(points) > 0 else None
        )


class CADCatalog:
    _managers = {}
    _all = None

    @classmethod
    def register(
        cls,
        dataset_name: str,
        cad_db_file: str,
        scene_file: str,
        point_file: str,
        grid_file: str
    ):
        if dataset_name in cls._managers:
            raise RuntimeError(
                'CAD models for {} are already registered'.format(dataset_name)
            )
        cls._managers[dataset_name] = CADManager(
            db_file=cad_db_file,
            scene_file=scene_file,
            point_file=point_file,
            grid_file=grid_file
        )

    @classmethod
    def deregister(cls, dataset_name: str):
        del cls._managers[dataset_name]

    @classmethod
    def get(cls, dataset_name: str) -> CADManager:
        return cls._managers[dataset_name]

    @classmethod
    def all(cls, reset_cache=False):
        use_cache = not reset_cache
        if cls._all is None or not use_cache:
            cls._all = CADManager.merge(*cls._managers.values())
        return cls._all


def register_cads(
    dataset_name: str,
    cad_db_file: str,
    scene_file: Optional[str] = None,
    point_file: Optional[str] = None,
    grid_file: Optional[str] = None
):
    CADCatalog.register(
        dataset_name,
        cad_db_file,
        scene_file,
        point_file=point_file,
        grid_file=grid_file
    )


def deregister_cads(dataset_name: str):
    CADCatalog.deregister(dataset_name)
