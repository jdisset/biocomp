from sqlmodel import Field, SQLModel, create_engine, Session, select
from sqlalchemy.orm import registry
from typing import Optional


class PartsDB(SQLModel, registry=registry()):
    pass


# SUPER HACKY but waiting for sqlmodel to fix serialization_alias support:
ALIASES = {"utr5": "5'UTR", "utr3": "3'UTR", "uid": "UID"}


class Category(PartsDB, table=True):
    name: str = Field(primary_key=True)
    transcripted: bool
    translated: bool


def int_or_none(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


class Part(PartsDB, table=True):
    name: str = Field(primary_key=True)
    category: str = Field(foreign_key="category.name")

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
    utr5: Optional[str] = Field(
        default=None, foreign_key="l0.id", sa_column_kwargs={"name": "5'UTR"}, alias="5'UTR"
    )
    gene: Optional[str] = Field(default=None, foreign_key="l0.id")
    utr3: Optional[str] = Field(
        default=None, foreign_key="l0.id", sa_column_kwargs={"name": "3'UTR"}, alias="3'UTR"
    )
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
    output_category: str
    parameter_list: str


class Sequestron(PartsDB, table=True):
    id: int = Field(primary_key=True)
    type: str = Field(foreign_key="sequestrontype.name")
    negative_part: str = Field(foreign_key="part.name")
    positive_part: str = Field(foreign_key="part.name")
    output_part: str


def get_all_parts_from_database(db_url: str):
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
        "categories": categories,
        "parts": parts,
        "L0s": L0s,
        "L1s": L1s,
        "L2s": L2s,
        "sequestron_types": sequestron_types,
        "sequestrons": sequestrons,
    }
