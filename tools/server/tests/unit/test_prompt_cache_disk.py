import os
import glob
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()

PROMPT_A = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers."
)

PROMPT_B = (
    "The little girl had a red ball. She liked to play with it in the park "
    "every day with her dog, who was big and brown and always very happy."
)


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.slot_save_path = tempfile.mkdtemp()
    server.cache_disk = -1
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)


def complete(prompt: str) -> int:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]["prompt_n"]


def disk_entries() -> list[str]:
    return glob.glob(os.path.join(server.slot_save_path, "*.pcache-*"))


def test_prompt_cache_disk_restart():
    global server
    server.start()

    n_a = complete(PROMPT_A)
    n_b = complete(PROMPT_B)
    assert n_a > 20 and n_b > 20

    # exit writes the slot and the RAM entries to disk
    server.stop()
    assert len(disk_entries()) == 2

    server.start()

    # both conversations come back from disk, only the tail is processed
    assert complete(PROMPT_A) < 5
    assert complete(PROMPT_B) < 5

    with open(server.log_path) as f:
        log = f.read()
    assert log.count("tokens from disk") >= 2


def test_prompt_cache_disk_eviction():
    global server
    # RAM cache too small for two entries, so the older one is moved to disk
    server.cache_ram = 1
    server.n_ctx = 2048
    long_a = (PROMPT_A + " ") * 12
    long_b = (PROMPT_B + " ") * 12
    long_c = "One day a cat and a dog went to the market to buy some fish. " * 12
    server.start()

    complete(long_a)
    complete(long_b)
    complete(long_c)

    with open(server.log_path) as f:
        assert "to disk" in f.read()
    assert len(disk_entries()) >= 1

    # A was pushed out of RAM by B, and still restores without a full prefill
    assert complete(long_a) < 5


def test_prompt_cache_disk_key_mismatch():
    global server
    server.start()

    n_a = complete(PROMPT_A)
    server.stop()
    assert len(disk_entries()) == 1

    # entries saved with other settings are removed instead of restored
    server.n_ctx = 1024
    server.start()
    assert len(disk_entries()) == 0
    assert complete(PROMPT_A) == n_a


def test_prompt_cache_disk_shared():
    global server
    # 1 MiB holds only one of these entries
    server.cache_disk = 1
    server.cache_disk_shared = True
    server.n_ctx = 2048
    server.start()
    complete((PROMPT_A + " ") * 12)
    server.stop()
    files_a = disk_entries()
    assert len(files_a) == 1

    # a server with other settings keeps the entry, then removes it to make room in the shared limit
    server.n_ctx = 4096
    server.start()
    assert disk_entries() == files_a
    complete((PROMPT_B + " ") * 12)
    server.stop()
    files = disk_entries()
    assert len(files) == 1 and files != files_a


def test_prompt_cache_disk_mtmd_text_only():
    global server
    # text-only prompts on a server with an mmproj loaded are written and restored like any other
    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.slot_save_path = tempfile.mkdtemp()
    server.cache_disk = -1
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    server.start()

    n_a = complete(PROMPT_A)
    n_b = complete(PROMPT_B)
    assert n_a > 20 and n_b > 20

    server.stop()
    assert len(disk_entries()) == 2

    # only the tail is processed (a few more tokens than tinyllama2, sliding window attention)
    server.start()
    assert complete(PROMPT_A) < n_a // 4
    assert complete(PROMPT_B) < n_b // 4

    with open(server.log_path) as f:
        assert f.read().count("tokens from disk") >= 2
