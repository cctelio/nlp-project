from src.tokenization.atomwise import regex_tokenize_smiles
from src.tokenization.base import AtomwiseSmilesRepresentation, SmilesPESmilesRepresentation


def test_atomwise_regex_roundtrip():
    smiles = "CC(=O)Cl"
    tokens = regex_tokenize_smiles(smiles, strict=True)
    assert tokens == ["C", "C", "(", "=", "O", ")", "Cl"]
    assert "".join(tokens) == smiles


def test_atomwise_regex_non_strict_preserves_unmatched_characters():
    smiles = "C%"
    tokens = regex_tokenize_smiles(smiles, strict=False)
    assert "".join(tokens) == smiles


def test_atomwise_representation_detokenizes_generated_text():
    representation = AtomwiseSmilesRepresentation()
    encoded = representation.encode_smiles("CCO")
    assert encoded == "C C O"
    assert representation.decode_generation(encoded) == "CCO"


def test_smilespe_learns_merges():
    representation = SmilesPESmilesRepresentation.train(["CCO", "CCN", "CCCl"], vocab_size=8, min_frequency=2)
    assert representation.merges
    encoded = representation.encode_smiles("CCO")
    assert representation.decode_generation(encoded) == "CCO"
