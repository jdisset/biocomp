from sqlmodel import Field, SQLModel, Relationship
from sqlmodel import SQLModel, create_engine, Session, select
from sqlmodel._compat import SQLModelConfig
from typing import List, Optional
from sqlalchemy import Column, JSON
from biocomp.library import PartsLibrary
import pandas as pd

class PartsDB(SQLModel):
    pass

# SUPER HACKY but waiting for sqlmodel to fix serialization_alias support:
ALIASES = { 'utr5': "5'UTR", 'utr3': "3'UTR", "uid": "UID" }

class Category(PartsDB, table=True):
    name: str = Field(primary_key=True)
    transcripted: int
    translated: int

class Part(PartsDB, table=True):
    name: str = Field(primary_key=True)
    category: str = Field(foreign_key="category.name")
    uid: Optional[int] = Field(default=None, sa_column_kwargs={"name": "UID"}, alias="UID")

    class Config:
        # SUPER HACKY but waiting for sqlmodel to fix serialization_alias support...
        alias_generator = lambda field_name: ALIASES.get(field_name, field_name)


class L0(PartsDB, table=True):
    id: str = Field(primary_key=True)
    notes: Optional[str] = None
    constructed: bool
    backbone: str
    part_1: Optional[str] = Field(default=None, foreign_key="part.name")
    part_2: Optional[str] = Field(default=None, foreign_key="part.name")
    part_3: Optional[str] = Field(default=None, foreign_key="part.name")
    part_4: Optional[str] = Field(default=None, foreign_key="part.name")
    part_5: Optional[str] = Field(default=None, foreign_key="part.name")
    part_6: Optional[str] = Field(default=None, foreign_key="part.name")


class L1(PartsDB, table=True):
    id: str = Field(primary_key=True)
    notes: Optional[str] = None
    constructed: bool
    backbone: str
    insulator: Optional[str] = Field(default=None, foreign_key="l0.id")
    promoter: Optional[str] = Field(default=None, foreign_key="l0.id")
    utr5: Optional[str] = Field(default=None, foreign_key="l0.id", sa_column_kwargs={"name": "5'UTR"}, alias="5'UTR")
    gene: Optional[str] = Field(default=None, foreign_key="l0.id")
    utr3: Optional[str] = Field(default=None, foreign_key="l0.id", sa_column_kwargs={"name": "3'UTR"}, alias="3'UTR")
    terminator: Optional[str] = Field(default=None, foreign_key="l0.id")

    class Config:
        # SUPER HACKY but waiting for sqlmodel to fix serialization_alias support...
        alias_generator = lambda field_name: ALIASES.get(field_name, field_name)

class L2(PartsDB, table=True):
    id: str = Field(primary_key=True)
    notes: Optional[str] = None
    constructed: bool
    backbone: str
    slot_1: Optional[str] = Field(default=None, foreign_key="l1.id")
    slot_2: Optional[str] = Field(default=None, foreign_key="l1.id")
    slot_3: Optional[str] = Field(default=None, foreign_key="l1.id")
    slot_4: Optional[str] = Field(default=None, foreign_key="l1.id")
    slot_5: Optional[str] = Field(default=None, foreign_key="l1.id")
    slot_6: Optional[str] = Field(default=None, foreign_key="l1.id")


class SequestronType(PartsDB, table=True):
    name: str = Field(primary_key=True)
    negative_category: str = Field(foreign_key="category.name")
    positive_category: str = Field(foreign_key="category.name")
    negative_level: str
    positive_level: str
    output_level: str
    output_side: str
    # output_category: List[str] = Field(sa_column=Column(JSON))
    # parameter_list: List[str] = Field(sa_column=Column(JSON))
    output_category: str
    parameter_list: str


class Sequestron(PartsDB, table=True):
    id: int = Field(primary_key=True)
    type: str = Field(foreign_key="sequestrontype.name")
    negative_part: str = Field(foreign_key="part.name")
    positive_part: str = Field(foreign_key="part.name")
    # output_part: List[str] = Field(sa_column=Column(JSON))
    output_part: str



def getAllPartsFromDatabase(db_url: str):
    engine = create_engine(db_url)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        categories = session.exec(select(Category)).all()
        parts = session.exec(select(Part)).all()
        L0s = session.exec(select(L0)).all()
        L1s = session.exec(select(L1)).all()
        L2s = session.exec(select(L2)).all()
        sequestron_types = session.exec(select(SequestronType)).all()
        sequestrons = session.exec(select(Sequestron)).all()

    return {
        'categories': categories,
        'parts': parts,
        'L0s': L0s,
        'L1s': L1s,
        'L2s': L2s,
        'sequestron_types': sequestron_types,
        'sequestrons': sequestrons
    }


def buildLibFromDatabase(db_url: str):
    parts = getAllPartsFromDatabase(db_url)
    # first need to turn everything into pandas dataframes
    parts_dict = {}
    for key, value in parts.items():
        # we also need to use the primary key as the index
        pk_field_name = value[0].__table__.primary_key.columns.keys()[0]
        as_dict = [x.model_dump(by_alias=True) for x in value]
        df = pd.DataFrame(as_dict)
        df.set_index(pk_field_name, inplace=True)
        parts_dict[key] = df

    lib = PartsLibrary(
        parts_dict['parts'], parts_dict['L0s'], parts_dict['L1s'], parts_dict['L2s'], parts_dict['categories'], parts_dict['sequestrons'], parts_dict['sequestron_types']
    )
    return lib
