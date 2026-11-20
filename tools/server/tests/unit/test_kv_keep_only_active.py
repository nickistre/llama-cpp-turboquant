import os
import tempfile
import threading
import time
import pytest
from utils import *

server = ServerPreset.tinyllama2()

class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


# idle slot cleared on launch should restore from cache-ram
def test_clear_and_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    # verify feature is enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    original_prompt_n = res.body["timings"]["prompt_n"]

    # Slot 0 is the only slot with KV — should NOT be cleared
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()

    # Launching slot 1 clears idle slot 0
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.drain()

    # Re-send same prompt — should restore from cache-ram
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "updating prompt cache" in log.drain()
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < original_prompt_n

    # Follow-up — slot 0 kept its KV, no clearing needed
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def test_disabled_with_flag():
    global server
    server.no_cache_idle_slots = True
    server.start()
    log = LogReader(server.log_path)

    # Feature should not be enabled
    assert "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__" not in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # Request on different slot — should NOT trigger clearing
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" not in log.drain()


def token_ids(word, n):
    res = server.make_request("POST", "/tokenize", data={"content": word})
    assert res.status_code == 200
    toks = res.body["tokens"]
    return (toks * ((n // len(toks)) + 1))[:n]


def stream_in_thread(prompt_tokens, n_predict, started=None, extra=None):
    box = {"chunks": []}

    def run():
        data = {
            "prompt": prompt_tokens,
            "n_predict": n_predict,
            "ignore_eos": True,
            "cache_prompt": True,
            "stream": True,
        }
        if extra:
            data.update(extra)
        try:
            for i, c in enumerate(server.make_stream_request("POST", "/completion", data=data)):
                if i == 0 and started is not None:
                    started.set()
                box["chunks"].append(c)
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=run)
    t.start()
    return t, box


def n_token_chunks(box):
    return len([c for c in box["chunks"] if c.get("tokens")])


# hold the caller until the stream in box has delivered n tokens
def await_decode_progress(box, n, timeout=120):
    deadline = time.time() + timeout
    while n_token_chunks(box) < n and time.time() < deadline:
        time.sleep(0.001)
    assert n_token_chunks(box) >= n


def assert_clean_stream(box, n_predict):
    assert "error" not in box
    assert not [c for c in box["chunks"] if "error" in c]
    assert not [c for c in box["chunks"] if not isinstance(c, dict)]
    assert n_token_chunks(box) == n_predict
    # the reported count never restarts across a park
    seq = [c["tokens_predicted"] for c in box["chunks"] if c.get("tokens")]
    assert seq == list(range(1, n_predict + 1))
    assert box["chunks"][-1]["tokens_predicted"] == n_predict


# two sessions in a 1280-cell pool: A (220 + 1000 tokens) alone fits, but A and B (220 + 600 tokens) together do not
# B starts once A is 40 tokens in, so the pool runs out while both sessions decode
def run_two_sessions(prompt_b_len=220, n_predict_b=600, extra=None):
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 1000, started, extra)
    assert started.wait(30)
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread(token_ids("dragon", prompt_b_len), n_predict_b, extra=extra)
    ta.join(120)
    tb.join(120)
    return box_a, box_b


# stock behavior: the decode that overflows the pool fails every session
def test_decode_overflow_killall_flag_off():
    global server
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    box_a, box_b = run_two_sessions()

    for box in (box_a, box_b):
        errors = [c["error"]["message"] for c in box["chunks"] if "error" in c]
        assert errors and "context size has been exceeded" in errors[0].lower()
    assert "Context size has been exceeded." in log.drain()


@pytest.mark.parametrize("policy", ["newest", "largest", "least-output"])
def test_decode_overflow_park_and_resume(policy):
    global server
    server.decode_overflow_policy = policy
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    box_a, box_b = run_two_sessions()

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_POLICY_ENABLED__" in content
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_DECODE_OVERFLOW_RESUME__" in content
    assert "Context size has been exceeded." not in content

    assert_clean_stream(box_a, 1000)
    assert_clean_stream(box_b, 600)

    # the resume reports like an uninterrupted run: only the client prompt was processed, nothing came from the cache
    for box in (box_a, box_b):
        final = box["chunks"][-1]
        assert final["timings"]["prompt_n"] == 220
        assert final["timings"]["cache_n"] == 0
        assert final["timings"]["predicted_n"] == final["tokens_predicted"]


# the parked sampler keeps the grammar parse state: a reset grammar would open a second brace after the resume
def test_decode_overflow_grammar_parked_and_resumed():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    box_a, box_b = run_two_sessions(extra={"grammar": 'root ::= "{" [a]+'})

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "Context size has been exceeded." not in content

    assert_clean_stream(box_a, 1000)
    assert_clean_stream(box_b, 600)

    for box in (box_a, box_b):
        text = "".join(c.get("content", "") for c in box["chunks"])
        assert text.startswith("{")
        assert text.count("{") == 1
        assert set(text[1:]) == {"a"}


# non-streaming: the response holds the output from before and after the park
def test_decode_overflow_parked_nonstreaming_complete():
    global server
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    prompt_b = token_ids("dragon", 220)
    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 1000, started)
    assert started.wait(30)
    await_decode_progress(box_a, 40)

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_b,
        "n_predict": 600,
        "ignore_eos": True,
        "cache_prompt": True,
        "return_tokens": True,
    }, timeout=120)
    ta.join(120)

    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in log.drain()
    assert_clean_stream(box_a, 1000)

    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 600
    assert len(res.body["tokens"]) == 600
    assert res.body["tokens_evaluated"] == 220
    assert res.body["timings"]["prompt_n"] == 220
    # the echo is the client prompt only, without the output the parked session resumed from
    assert res.body["prompt"].strip() == server.make_request("POST", "/detokenize", data={"tokens": prompt_b}).body["content"].strip()


# a reserve larger than the pool holds back every new session, and with all slots idle it fails loudly
def test_decode_overflow_reserve_blocks_and_fails_loud():
    global server
    server.decode_overflow_policy = "newest"
    server.decode_overflow_reserve = 2000
    server.n_ctx = 1280
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "n_predict": 4,
    })
    assert res.status_code != 200
    assert "decode-overflow reserve" in res.body["error"]["message"]
    assert "no room even after clearing idle slots" in log.drain()


@pytest.mark.parametrize("setup,reason", [
    ({"cache_ram": 0}, "requires --cache-ram"),
    ({"kv_unified": False}, "requires --kv-unified"),
])
def test_decode_overflow_policy_disabled(setup, reason):
    global server
    server.decode_overflow_policy = "newest"
    for k, v in setup.items():
        setattr(server, k, v)
    server.start()
    log = LogReader(server.log_path)

    assert f"--decode-overflow-policy {reason}, disabling" in log.drain()

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_PROMPT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


# a restore that fails for lack of cells moves the idle slots into the prompt cache and retries once
@pytest.mark.parametrize("n_slots,n_ctx,cache_ram,n_tokens_a,n_tokens_b", [
    (2, 256, 100, 180, 150),
    (3, 256, 100, 180, 150),
    # both prompts fit in KV, but their states do not fit together in RAM
    (2, 2048, 1, 1200, 700),
])
def test_restore_retry_after_clearing_idle_slots(n_slots, n_ctx, cache_ram, n_tokens_a, n_tokens_b):
    global server
    server.n_slots = n_slots
    server.n_ctx = n_ctx
    server.cache_ram = cache_ram
    server.override_kv = f"llama.context_length=int:{n_ctx}"
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/tokenize", data={"content": "Hello world"})
    assert res.status_code == 200
    tokens = res.body["tokens"]
    prompt_a = [tokens[0]] * n_tokens_a
    prompt_b = [tokens[-1]] * n_tokens_b

    for prompt, n_cached in [
        (prompt_a, 0),
        (prompt_a + [tokens[0]] * 10, n_tokens_a),
        (prompt_b, 0),
        (prompt_a + [tokens[0]] * 20, n_tokens_a + 10),
        (prompt_b + [tokens[-1]] * 10, n_tokens_b),
        (prompt_a + [tokens[0]] * 30, n_tokens_a + 20),
    ]:
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "n_predict": 1,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        assert res.body["timings"]["cache_n"] >= n_cached
        assert res.body["timings"]["prompt_n"] <= len(prompt) - n_cached

    if n_ctx == 256:
        content = log.drain()
        assert "retrying prompt restore after clearing" in content
        assert "__TEST_TAG_CACHE_IDLE_SLOT__" in content


# the cached state holds generated tokens too, so the restore needs cells for more than the returning prompt
def test_restore_state_larger_than_prompt():
    global server
    server.n_ctx = 256
    server.override_kv = "llama.context_length=int:256"
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/tokenize", data={"content": "Hello world"})
    assert res.status_code == 200
    tokens = res.body["tokens"]
    prompt_a = [tokens[0]] * 250
    prompt_b = [tokens[-1]] * 10

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "n_predict": 2,
        "ignore_eos": True,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # launching b moves the state of a into the prompt cache, and b keeps its cells after completing
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_b,
        "n_predict": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # at most 245 cells are free while the state of b is resident, and the state of a needs 252
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a + [tokens[0]] * 2,
        "n_predict": 2,
        "ignore_eos": True,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "retrying prompt restore after clearing" in log.drain()
    assert res.body["timings"]["cache_n"] >= len(prompt_a)
    assert res.body["timings"]["prompt_n"] <= 2


# an active decode holds the cells, so the restore waits for the slot to be released instead of processing the prompt again
def test_defer_restore_until_cells_free():
    global server
    server.n_ctx = 256
    server.override_kv = "llama.context_length=int:256"
    server.n_threads = 1
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/tokenize", data={"content": "Hello world"})
    assert res.status_code == 200
    tokens = res.body["tokens"]
    prompt_a = [tokens[0]] * 220
    prompt_b = [tokens[-1]] * 40

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "n_predict": 2,
        "ignore_eos": True,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    stream_first = threading.Event()
    stream_done = threading.Event()

    def run_stream():
        try:
            for _ in server.make_stream_request("POST", "/completion", data={
                "prompt": prompt_b,
                "n_predict": 160,
                "ignore_eos": True,
                "cache_prompt": True,
                "stream": True,
            }):
                stream_first.set()
        finally:
            stream_done.set()

    threading.Thread(target=run_stream).start()
    assert stream_first.wait(30)

    # at most 215 cells are free while b decodes, and the state of a needs 222
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a + [tokens[0]] * 2,
        "n_predict": 2,
        "ignore_eos": True,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert stream_done.wait(5)

    content = log.drain()
    assert "prompt restore failed, deferring task until a slot is released" in content
    assert res.body["timings"]["cache_n"] >= len(prompt_a)
    assert res.body["timings"]["prompt_n"] <= 2


IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"


# the same pool setup as the fixture, on a multimodal model (an mmproj is loaded)
def multimodal_server():
    global server
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    log_path = server.log_path
    server = ServerPreset.tinygemma3()
    server.n_slots = 2
    server.n_predict = 4
    server.n_batch = 512
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.kv_unified = True
    server.debug = True
    server.log_path = log_path
    server.decode_overflow_policy = "newest"
    server.n_ctx = 1280
    return server


# parking is gated on the slot's content: text-only sessions on a multimodal model park and resume
def test_decode_overflow_park_text_only_on_multimodal():
    multimodal_server().start()
    log = LogReader(server.log_path)

    box_a, box_b = run_two_sessions()

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "__TEST_TAG_DECODE_OVERFLOW_RESUME__" in content
    assert "Context size has been exceeded." not in content

    assert_clean_stream(box_a, 1000)
    assert_clean_stream(box_b, 600)


# a session holding an image is never parked: with "newest", the newer image session would be the victim,
# so the older text-only session is parked instead
def test_decode_overflow_never_parks_image_session():
    import base64
    import requests

    multimodal_server().start()
    log = LogReader(server.log_path)

    img = base64.b64encode(requests.get(IMG_URL_CAT).content).decode("utf-8")

    started = threading.Event()
    ta, box_a = stream_in_thread(token_ids("knight", 220), 1000, started)
    assert started.wait(30)
    await_decode_progress(box_a, 40)
    tb, box_b = stream_in_thread({
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [img],
    }, 600)
    ta.join(120)
    tb.join(120)

    content = log.drain()
    assert "__TEST_TAG_DECODE_OVERFLOW_PARK__" in content
    assert "Context size has been exceeded." not in content

    assert_clean_stream(box_a, 1000)
    assert_clean_stream(box_b, 600)
