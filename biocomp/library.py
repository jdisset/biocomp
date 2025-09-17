from typing import Dict, List, Tuple, Optional
import pandas as pd
from pydantic import BaseModel
import json5
from biocomp.models import get_all_parts_from_database
from biocomp.utils import BIOCOMP_ROOT_PATH, flatten
from pathlib import Path
import os


def j5loads(x):
    try:
        return json5.loads(x)
    except Exception as e:
        print(f"Error loading {x}: {e}")
        return x


def decode_json(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: j5loads(str(x)))
    return df


L1_SLOT_KEYS = ["promoter", "5'UTR", "gene", "insulator", "3'UTR", "terminator"]


class PartsLibrary(BaseModel):
    parts: pd.DataFrame
    L0s: pd.DataFrame
    L1s: pd.DataFrame
    L2s: pd.DataFrame
    categories: pd.DataFrame
    sequestrons: pd.DataFrame
    sequestron_types: pd.DataFrame
    pc: Optional[pd.DataFrame] = None
    seqs: Optional[pd.DataFrame] = None
    _l0_cache: Optional[dict] = None
    _l1_cache: Optional[dict] = None

    model_config = {"arbitrary_types_allowed": True}

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
        self.seqs = decode_json(self.seqs, ["output_part", "output_category"])
        self.seqs["enabled"] = True

        # Cache L0 and L1 parts for performance
        self._l0_cache = {}
        self._l1_cache = {}

    def disable_all_sequestrons(self) -> None:
        """Disable all sequestrons"""
        self.seqs["enabled"] = False

    def enable_all_sequestrons(self) -> None:
        """Enable all sequestrons"""
        self.seqs["enabled"] = True

    def enable_sequestrons(self, sequestron_types: List[str]) -> None:
        """Enable specific sequestron types"""
        self.seqs.loc[self.seqs.type.isin(sequestron_types), "enabled"] = True

    def disable_sequestrons(self, sequestron_types: List[str]) -> None:
        """Disable specific sequestron types"""
        self.seqs.loc[self.seqs.type.isin(sequestron_types), "enabled"] = False

    def set_enabled_sequestrons(self, sequestron_types: List[str]) -> None:
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

    def add_sequestron(self, dic: Dict) -> None:
        """Add a new sequestron"""
        self.sequestrons = pd.concat([self.sequestrons, pd.DataFrame([dic])], ignore_index=True)
        self.seqs = self.sequestrons.merge(self.sequestron_types, left_on="type", right_index=True)
        self.seqs = decode_json(self.seqs, ["output_part", "output_category"])

    def get_rna(self, dna: str) -> Tuple[str, ...]:
        """Get RNA for given DNA"""
        d = self.pc.loc[dna]
        return tuple(d[d.transcripted == 1].index)

    def get_prt(self, dna: str) -> Tuple[str, ...]:
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


DEFAULT_LIB_PATH = Path(BIOCOMP_ROOT_PATH).expanduser() / "partsdb.sqlite"
DEFAULT_LIB_PATH = f"sqlite:///{DEFAULT_LIB_PATH}"
if "BIOCOMP_PARTS_DB" in os.environ:
    DEFAULT_LIB_PATH = Path(os.environ["BIOCOMP_PARTS_DB"]).expanduser().resolve()


def build_lib_from_database(db_url: str) -> PartsLibrary:
    parts = get_all_parts_from_database(db_url)

    if len(parts["parts"]) == 0:
        raise ValueError("No parts found in database")

    parts_dict = {}
    for key, value in parts.items():
        # Convert to pandas DataFrame using primary key as index
        pk_field_name = value[0].__table__.primary_key.columns.keys()[0]
        as_dict = [x.model_dump(by_alias=True) for x in value]
        df = pd.DataFrame(as_dict)
        df.set_index(pk_field_name, inplace=True)
        parts_dict[key] = df

    return PartsLibrary(**parts_dict)


def load_lib(lib_path=DEFAULT_LIB_PATH):
    if "lib_path" not in load_lib.__dict__ or load_lib.lib_path != lib_path:
        load_lib.lib = build_lib_from_database(lib_path)
        load_lib.lib_path = lib_path
    return load_lib.lib


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
