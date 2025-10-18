from pydantic import BaseModel


class PartEmbedding(BaseModel):
    """Defines a biological part embedding with its metadata"""

    name: str  # "tc_rate", "tl_rate"
    part_categories: list[str]  # ["promoter"], ["uORF_group"]
    default_part: str
    available_parts: list[str]


# Single source of truth for all part embeddings
PART_EMBEDDINGS = [
    PartEmbedding(
        name="tc_rate",
        part_categories=["promoter"],
        default_part="hEF1a",
        available_parts=["hEF1a"],
    ),
    PartEmbedding(
        name="tl_rate",
        part_categories=["uORF_group"],
        default_part="00_empty_tc",
        available_parts=[
            "00_empty_tc",
            "1w_uORF",
            "1x_uORF",
            "2x_uORF",
            "3x_uORF",
            "4x_uORF",
            "5x_uORF",
            "6x_uORF",
            "8x_uORF",
            "9x_uORF",
            "10x_uORF",
            "11x_uORF",
            "12x_uORF",
        ],
    ),
]

# Derived lookup dicts (built once at import)
EMBEDDINGS_BY_NAME = {e.name: e for e in PART_EMBEDDINGS}

EMBEDDINGS_BY_CATEGORY = {}
for emb in PART_EMBEDDINGS:
    for cat in emb.part_categories:
        EMBEDDINGS_BY_CATEGORY[cat] = emb
