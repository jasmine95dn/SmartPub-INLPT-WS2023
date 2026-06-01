"""
Stub all heavy ML / cloud dependencies before any source module is imported.
These stubs run at conftest import time, before pytest collects tests.
"""
import sys
from unittest.mock import MagicMock

# torch: configure device selection so CPU path is taken
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
_torch.device = MagicMock(side_effect=lambda x: x)
_torch.bfloat16 = "bfloat16"

# tqdm: pass the iterable through unchanged so for-loops execute normally
_tqdm_mod = MagicMock()
_tqdm_mod.tqdm = lambda iterable, **kwargs: iterable

_STUBS = {
    "torch": _torch,
    "torch.cuda": _torch.cuda,
    "transformers": MagicMock(),
    "pinecone": MagicMock(),
    "langchain": MagicMock(),
    "langchain.chains": MagicMock(),
    "langchain.chains.combine_documents": MagicMock(),
    "langchain.embeddings": MagicMock(),
    "langchain.embeddings.huggingface": MagicMock(),
    "langchain.vectorstores": MagicMock(),
    "langchain.llms": MagicMock(),
    "langchain.text_splitter": MagicMock(),
    "langchain_core": MagicMock(),
    "langchain_core.prompts": MagicMock(),
    "langchain_huggingface": MagicMock(),
    "langchain_pinecone": MagicMock(),
    "langchain_text_splitters": MagicMock(),
    "sentence_transformers": MagicMock(),
    "sklearn": MagicMock(),
    "sklearn.metrics": MagicMock(),
    "sklearn.metrics.pairwise": MagicMock(),
    "tqdm": _tqdm_mod,
    "dotenv": MagicMock(),
    "pandas": MagicMock(),
    "numpy": MagicMock(),
}

for _name, _mock in _STUBS.items():
    sys.modules[_name] = _mock
