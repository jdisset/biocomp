from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Annotated
from . import utils as ut
import json5


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
