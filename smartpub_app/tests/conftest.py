import pytest


SAMPLE_TRIPLETS = [
    {"head": "COVID-19", "type": "causes", "tail": "pneumonia"},
    {"head": "aspirin", "type": "treats", "tail": "headache"},
]

SAMPLE_ENTRY = {
    "title_entity_relation": {
        "pmid": "12345",
        "authors": "Smith J, Doe A",
        "year": "2023",
        "triplets": SAMPLE_TRIPLETS,
    },
    "abstract_entity_relation": {
        "triplets": [{"head": "virus", "type": "infects", "tail": "cell"}],
    },
}


@pytest.fixture
def sample_triplets():
    return list(SAMPLE_TRIPLETS)


@pytest.fixture
def sample_entry():
    return SAMPLE_ENTRY
