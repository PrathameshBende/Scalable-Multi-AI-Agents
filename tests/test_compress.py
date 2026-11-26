import numpy as np
import pytest

from scalable_agents.compress import mean_pool, parse_hellaswag, parse_perplexity


def test_mean_pool_ignores_padding_and_normalises():
    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]])  # last token is padding
    mask = np.array([[1, 1, 0]])
    out = mean_pool(hidden, mask)
    assert np.allclose(out, [[1.0, 0.0]])  # mean (2, 0) normalised


PPL_SAMPLE = """perplexity: calculating perplexity over 655 chunks, n_ctx=512, batch_size=2048
[1]9.4105,[2]12.0331,[3]12.8823,
Final estimate: PPL = 13.8964 +/- 0.09879
"""

HS_SAMPLE = """hellaswag_score : calculating HellaSwag score over 400 tasks.
task\tacc_norm\t95% confidence interval
1\t100.00000000%\t[20.6549%, 100.0000%]
2\t50.00000000%\t[9.4529%, 90.5471%]
400\t47.25000000%\t[42.4064%, 52.1440%]

llama_perf_context_print:        load time =     812.52 ms
"""


def test_parse_perplexity():
    assert parse_perplexity(PPL_SAMPLE) == {"ppl": 13.8964, "ppl_stderr": 0.09879}
    with pytest.raises(ValueError):
        parse_perplexity("nothing")


def test_parse_hellaswag_takes_last_line():
    r = parse_hellaswag(HS_SAMPLE)
    assert r["tasks"] == 400 and r["acc"] == 47.25
    assert r["ci_low"] == 42.4064 and r["ci_high"] == 52.144
    assert parse_hellaswag("10\t45.5")["acc"] == 45.5  # older format without CI
