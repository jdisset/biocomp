# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
from typing import Any
import pandas as pd
import pickle
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from biocomp.models import (
    Part,
    L0,
    L1,
    L2,
    Category,
    Sequestron,
    SequestronType,
    PartsRecord,
)
from biocomp.utils import BIOCOMP_ROOT_PATH, flatten
from pathlib import Path
import os


L1_SLOT_KEYS = [
    "insulator",
    "promoter",
    "5'UTR",
    "gene",
    "3'UTR",
    "terminator",
]

L0_SLOT_KEYS = [f"part_{i}" for i in range(1, 7)]
L2_SLOT_KEYS = [f"slot_{i}" for i in range(1, 7)]


# Maps each PartsLibrary field to (pydantic class, primary key column).
# Single source of truth for table identity used by validators, dumpers, and FK checks.
PARTS_SCHEMA: dict[str, tuple[type[PartsRecord], str]] = {
    "categories": (Category, "name"),
    "parts": (Part, "name"),
    "L0s": (L0, "id"),
    "L1s": (L1, "id"),
    "L2s": (L2, "id"),
    "sequestron_types": (SequestronType, "name"),
    "sequestrons": (Sequestron, "id"),
}


def _records_to_indexed_df(records: list[Any], pk: str) -> pd.DataFrame:
    rows = []
    for r in records:
        if hasattr(r, "model_dump"):
            rows.append(r.model_dump(by_alias=True))
        elif isinstance(r, dict):
            rows.append(r)
        else:
            raise TypeError(f"PartsLibrary record must be SQLModel or dict, got {type(r)}")
    df = pd.DataFrame(rows)
    if pk in df.columns:
        df.set_index(pk, inplace=True)
    return df


class PartsLibrary(BaseModel):
    parts: pd.DataFrame
    L0s: pd.DataFrame
    L1s: pd.DataFrame
    L2s: pd.DataFrame
    categories: pd.DataFrame
    sequestrons: pd.DataFrame
    sequestron_types: pd.DataFrame
    # Computed views -- overwritten in model_post_init. Default to empty so type is non-Optional.
    pc: pd.DataFrame = Field(default_factory=pd.DataFrame)
    seqs: pd.DataFrame = Field(default_factory=pd.DataFrame)

    _l0_cache: dict = PrivateAttr(default_factory=dict)
    _l1_cache: dict = PrivateAttr(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator(
        "categories", "parts", "L0s", "L1s", "L2s", "sequestron_types", "sequestrons",
        mode="before",
    )
    @classmethod
    def _coerce_records(cls, v, info):
        # Accept the canonical DataFrame form (already indexed) or a list of records/dicts
        # produced by dracon !include + !each over per-id YAML files.
        if isinstance(v, pd.DataFrame):
            return v
        if isinstance(v, list | tuple):
            _, pk = PARTS_SCHEMA[info.field_name]
            return _records_to_indexed_df(list(v), pk)
        return v

    @model_validator(mode="after")
    def _check_fks(self):
        # Empty strings and NaN are slot-absent sentinels in the legacy schema; ignore them.
        def present(values):
            return [v for v in values if v not in ("", None) and not (isinstance(v, float) and pd.isna(v))]

        cat_idx = set(self.categories.index)
        part_idx = set(self.parts.index)
        l0_idx = set(self.L0s.index)
        l1_idx = set(self.L1s.index)
        st_idx = set(self.sequestron_types.index)

        def assert_subset(values, allowed, label):
            missing = [v for v in present(values) if v not in allowed]
            assert not missing, f"FK violation: {label} not in target table: {sorted(set(missing))[:5]}..."

        assert_subset(self.parts["category"], cat_idx, "parts.category")
        for slot in L0_SLOT_KEYS:
            if slot in self.L0s.columns:
                assert_subset(self.L0s[slot], part_idx, f"L0s.{slot}")
        for slot in L1_SLOT_KEYS:
            if slot in self.L1s.columns:
                assert_subset(self.L1s[slot], l0_idx, f"L1s.{slot}")
        for slot in L2_SLOT_KEYS:
            if slot in self.L2s.columns:
                assert_subset(self.L2s[slot], l1_idx, f"L2s.{slot}")
        assert_subset(self.sequestrons["type"], st_idx, "sequestrons.type")
        assert_subset(self.sequestrons["negative_part"], part_idx, "sequestrons.negative_part")
        assert_subset(self.sequestrons["positive_part"], part_idx, "sequestrons.positive_part")
        return self

    def to_records(self) -> dict[str, list[PartsRecord]]:
        """Inverse of construction: rebuild typed SQLModel records keyed by table name.

        Powers dracon dump (per-id YAML emission) and the sqlite cache writer.
        """
        out: dict[str, list[PartsRecord]] = {}
        for field, (model_cls, pk) in PARTS_SCHEMA.items():
            # Names of optional fields (by alias) -- only these get empty-string->absent cleanup.
            optional_aliases = {
                (f.alias or name)
                for name, f in model_cls.model_fields.items()
                if not f.is_required()
            }
            df = getattr(self, field)
            records = []
            for pk_value, row in df.iterrows():
                d = {pk: pk_value, **{k: v for k, v in row.to_dict().items() if k != pk}}
                cleaned = {}
                for k, v in d.items():
                    is_nan = isinstance(v, float) and pd.isna(v)
                    is_empty = v == "" or v is None
                    if k in optional_aliases and (is_nan or is_empty):
                        continue
                    if is_nan:
                        # Required field but NaN -- preserve as None so pydantic surfaces the issue.
                        cleaned[k] = None
                    else:
                        cleaned[k] = v
                records.append(model_cls.model_validate(cleaned))
            out[field] = records
        return out

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        """Initialize computed fields after validation"""
        # Filter out empty indices
        self.L0s = self.L0s.loc[self.L0s.index != ""]
        self.L1s = self.L1s.loc[self.L1s.index != ""]
        self.L2s = self.L2s.loc[self.L2s.index != ""]

        # Remove duplicates
        self.L0s = self.L0s[~self.L0s.index.duplicated(keep="first")]
        self.L1s = self.L1s[~self.L1s.index.duplicated(keep="first")]
        self.L2s = self.L2s[~self.L2s.index.duplicated(keep="first")]
        self.parts = self.parts[~self.parts.index.duplicated(keep="first")]

        # Create merged DataFrames
        self.pc = pd.merge(
            self.parts, self.categories, left_on="category", right_index=True, how="left"
        )

        self.seqs = self.sequestrons.merge(self.sequestron_types, left_on="type", right_index=True)
        self.seqs["enabled"] = True

    def disable_all_sequestrons(self) -> None:
        """Disable all sequestrons"""
        self.seqs["enabled"] = False

    def enable_all_sequestrons(self) -> None:
        """Enable all sequestrons"""
        self.seqs["enabled"] = True

    def enable_sequestrons(self, sequestron_types: list[str]) -> None:
        """Enable specific sequestron types"""
        self.seqs.loc[self.seqs.type.isin(sequestron_types), "enabled"] = True

    def disable_sequestrons(self, sequestron_types: list[str]) -> None:
        """Disable specific sequestron types"""
        self.seqs.loc[self.seqs.type.isin(sequestron_types), "enabled"] = False

    def set_enabled_sequestrons(self, sequestron_types: list[str]) -> None:
        """Set which sequestron types should be enabled"""
        self.disable_all_sequestrons()
        self.enable_sequestrons(sequestron_types)

    def get_enabled_sequestrons(self) -> pd.DataFrame:
        """Get all enabled sequestrons"""
        return self.seqs[self.seqs.enabled]

    def add_part(self, part: str, category: str) -> None:
        """Add a new part with its category"""
        self.parts.loc[part] = {"category": category}
        self.pc = pd.merge(
            self.parts, self.categories, left_on="category", right_index=True, how="left"
        )

    def add_sequestron(self, dic: dict) -> None:
        """Add a new sequestron"""
        self.sequestrons = pd.concat([self.sequestrons, pd.DataFrame([dic])], ignore_index=True)
        self.seqs = self.sequestrons.merge(self.sequestron_types, left_on="type", right_index=True)

    def get_rna(self, dna: str) -> tuple[str, ...]:
        """Get RNA for given DNA"""
        d = self.pc.loc[dna]
        return tuple(d[d.transcripted == 1].index)

    def get_prt(self, dna: str) -> tuple[str, ...]:
        """Get protein for given DNA"""
        d = self.pc.loc[dna]
        return tuple(d[d.translated == 1].index)

    def __str__(self) -> str:
        return f"""
        Parts & categories: \n{self.pc}\n,
        ------------------------------------------
        Enabled sequestrons: \n{self.get_enabled_sequestrons()}\n
        ------------------------------------------
        L0s: \n{self.L0s}\n
        ------------------------------------------
        L1s: \n{self.L1s}\n
        ------------------------------------------
        L2s: \n{self.L2s}\n
        """


def _default_lib_path() -> Path:
    env = os.environ.get("BIOCOMP_PARTS_DB")
    if env:
        return Path(env).expanduser().resolve()
    root = BIOCOMP_ROOT_PATH or "."
    return Path(root).expanduser() / "parts-db"


DEFAULT_LIB_PATH: Path = _default_lib_path()


def _build_dracon_loader():
    """Construct a DraconLoader pre-loaded with every parts schema type as a short tag.

    Single source of truth for the tag vocabulary used by both load_lib_from_yaml
    and dump_lib_to_yaml -- same loader can compose YAML *into* records and emit
    records back *out* under the same short names.
    """
    from dracon import DraconLoader, SymbolEntry, auto_symbol

    loader = DraconLoader()
    for model_cls in (Part, L0, L1, L2, Category, Sequestron, SequestronType, PartsLibrary):
        loader.context.define(SymbolEntry(model_cls.__name__, auto_symbol(model_cls)))
    return loader


def _is_cache_fresh(parts_dir: Path, cache_pickle: Path) -> bool:
    """Makefile-style freshness check: is the pickle newer than every yaml source?

    Bails on the first source newer than the cache -- ~3 ms for ~450 files vs
    ~15 ms for content-hashing. mtime+directory-mtime jointly catch edits,
    additions, and deletions (POSIX filesystems bump parent dir mtime on
    create/unlink).
    """
    if not cache_pickle.exists():
        return False
    cache_mt = cache_pickle.stat().st_mtime_ns
    for p in parts_dir.rglob("*"):
        if ".cache" in p.parts:
            continue
        # Track both file and directory mtimes: dir mtime changes on add/remove.
        if not (p.is_file() and p.suffix == ".yaml") and not p.is_dir():
            continue
        if p.stat().st_mtime_ns > cache_mt:
            return False
    return True


def load_lib_from_yaml(path: str | Path) -> PartsLibrary:
    """Load a PartsLibrary from a parts-db folder (containing index.yaml) or a single yaml file."""
    p = Path(path).expanduser()
    target = p / "index.yaml" if p.is_dir() else p
    if not target.exists():
        raise FileNotFoundError(f"No parts-db YAML at {target}")
    loader = _build_dracon_loader()
    return loader.load(str(target))


def dump_lib_to_yaml(lib: PartsLibrary, parts_dir: str | Path) -> None:
    """Materialise a PartsLibrary as a parts-db folder.

    Layout:
      parts_dir/primitives/{categories,parts,sequestron_types}.yaml  # flat lists
      parts_dir/{L0,L1,L2,sequestrons}/<id>.yaml                     # one per record
      parts_dir/index.yaml                                            # composer
    """
    parts_dir = Path(parts_dir).expanduser()
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "primitives").mkdir(exist_ok=True)
    for sub in ("L0", "L1", "L2", "sequestrons"):
        (parts_dir / sub).mkdir(exist_ok=True)

    loader = _build_dracon_loader()
    records = lib.to_records()

    # Primitives: one flat YAML list per table.
    for field, subdir_name in [
        ("categories", "primitives/categories.yaml"),
        ("parts", "primitives/parts.yaml"),
        ("sequestron_types", "primitives/sequestron_types.yaml"),
    ]:
        out_path = parts_dir / subdir_name
        out_path.write_text(loader.dump(records[field]))

    # Per-record files for L0/L1/L2/sequestrons. Filename = primary key; safe-ified.
    # Case-insensitive FS (macOS default) requires disambiguation when two ids differ
    # only by case (e.g. L1_eBFP2 vs L1_EBFP2).
    for field, subdir in [("L0s", "L0"), ("L1s", "L1"), ("L2s", "L2"), ("sequestrons", "sequestrons")]:
        seen_ci: dict[str, int] = {}
        for rec in records[field]:
            pk_value = str(getattr(rec, PARTS_SCHEMA[field][1]))
            base = _safe_filename(pk_value)
            key = base.lower()
            n = seen_ci.get(key, 0)
            seen_ci[key] = n + 1
            fname = f"{base}.yaml" if n == 0 else f"{base}__{n}.yaml"
            (parts_dir / subdir / fname).write_text(loader.dump(rec))

    (parts_dir / "index.yaml").write_text(_canonical_index_yaml())


def _canonical_index_yaml() -> str:
    """Return the canonical parts-db index template shipped with biocomp."""
    return (Path(__file__).parent / "config" / "parts_db_index.yaml").read_text()


def _safe_filename(pk: str) -> str:
    # Preserve the id as readably as possible; replace shell/path-hostile characters only.
    bad = "/\\:<>|\"?*"
    out = "".join("_" if c in bad else c for c in pk).strip(". ")
    return out or "_"


_LOAD_LIB_CACHE: dict[Any, PartsLibrary] = {}


def load_lib(lib_path=DEFAULT_LIB_PATH):
    """Polymorphic loader. Dispatches by path form:

      * directory or *.yaml -> compose from a parts-db folder (yaml is SSOT)
      * *.pickle            -> unpickle a serialised PartsLibrary

    A pickle cache living under <parts-db>/.cache/ is auto-used when its mtime
    is newer than every yaml source; the first edit, add, or remove invalidates
    it on the next load with no manual bookkeeping.
    """
    if lib_path in _LOAD_LIB_CACHE:
        return _LOAD_LIB_CACHE[lib_path]

    p_str = str(lib_path)
    p = Path(p_str).expanduser()
    lib: PartsLibrary
    if p.is_dir() or p_str.endswith(".yaml"):
        lib = _load_lib_from_yaml_cached(p)
    elif p_str.endswith(".pickle"):
        with open(p, "rb") as f:
            lib = pickle.load(f)
    else:
        raise ValueError(
            f"Cannot infer parts-db format from {p_str!r}. "
            "Expected a directory, *.yaml, or *.pickle."
        )

    _LOAD_LIB_CACHE[lib_path] = lib
    return lib


def _load_lib_from_yaml_cached(p: Path) -> PartsLibrary:
    # Single-file yaml: no caching layer (cheap enough); folder gets a pickle cache.
    if not p.is_dir():
        return load_lib_from_yaml(p)

    cache_dir = p / ".cache"
    cache_pickle = cache_dir / "library.pickle"

    if _is_cache_fresh(p, cache_pickle):
        try:
            with open(cache_pickle, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # fall through to rebuild on any unpickling problem

    lib = load_lib_from_yaml(p)
    cache_dir.mkdir(exist_ok=True)
    with open(cache_pickle, "wb") as f:
        pickle.dump(lib, f)
    return lib


def get_l0_parts(l0id, lib):
    if l0id not in lib.L0s.index:
        return None
    if l0id in lib._l0_cache:
        return lib._l0_cache[l0id]
    row = lib.L0s.loc[l0id]
    result = [row.get(f"part_{i}") for i in range(7) if row.get(f"part_{i}")]
    lib._l0_cache[l0id] = result
    return result


def get_l1_parts(l1id, lib):
    if l1id not in lib.L1s.index:
        return None
    if l1id in lib._l1_cache:
        return lib._l1_cache[l1id]
    row = lib.L1s.loc[l1id]
    result = flatten([get_l0_parts(row[k], lib) for k in L1_SLOT_KEYS if row[k]])
    lib._l1_cache[l1id] = result
    return result


def get_l1_from_l2(l2id, lib):
    if l2id not in lib.L2s.index:
        return None
    keys = [f"slot_{i}" for i in range(7)]
    return [lib.L2s.loc[l2id].get(k) for k in keys if lib.L2s.loc[l2id].get(k)]


## {{{                      --     Library Context     --


class LibraryContext:
    _current_lib = None

    @classmethod
    def set_library(cls, lib):
        cls._current_lib = lib

    @classmethod
    def get_library(cls):
        if cls._current_lib is None:
            return load_lib()
        return cls._current_lib

    @classmethod
    def with_library(cls, lib):
        """Context manager for temporarily setting a library"""

        class LibraryContextManager:
            def __init__(self, lib):
                self.lib = lib
                self.previous_lib = None

            def __enter__(self):
                self.previous_lib = LibraryContext._current_lib
                LibraryContext.set_library(self.lib)
                return self.lib

            def __exit__(self, exc_type, exc_val, exc_tb):
                LibraryContext.set_library(self.previous_lib)

        return LibraryContextManager(lib)


##────────────────────────────────────────────────────────────────────────────}}}
