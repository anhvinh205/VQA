from pathlib import Path

from src.data.vocab import PAD, UNK, Vocab, tokenize_words

def test_tokenize_words_lowercases_and_splits():
    tokens = tokenize_words("Is this a Creamy Soup?")
    assert tokens == ["is", "this", "a", "creamy", "soup", "?"]
    
def test_build_respects_min_freq():
    questions = ["cat dog", "cat bird", "cat"]
    vocab = Vocab.build(questions, min_freq=2)
    assert "cat" in vocab.token_to_idx
    assert "dog" not in vocab.token_to_idx  
    
def test_unknown_token_maps_to_unk():
    vocab = Vocab.build(["hello world"], min_freq=1)
    assert vocab["zzz_not_in_vocab"] == vocab[UNK]
    
def test_encode_pasds_and_truncates():
    vocab = Vocab.build(["a b c d e"], min_freq=1)
    short = vocab.encode("a b", max_seq_len=5)
    assert len(short) == 5
    assert short[-1] == vocab[PAD]
    
    long_ = vocab.encode("a b c d e", max_seq_len=3)
    assert len(long_) == 3
    
def test_save_and_load_roundtrip(tmp_path: Path):
    vocab = Vocab.build(["hello world", "hello there"], min_freq=1)
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = Vocab.load(path)
    assert loaded.token_to_idx == vocab.token_to_idx
    assert len(loaded) == len(vocab)    
    
    