from neurolens.ethics import evaluate, DARK_PATTERN_AMYG, DARK_PATTERN_NACC, AROUSAL_CEILING


def _scores(**overrides: float) -> dict[str, float]:
    base = {
        "FFA": 0.4, "V4": 0.4, "MT+": 0.4, "Hippocampus": 0.4,
        "PFC": 0.4, "ACC": 0.3, "Amygdala": 0.2, "Insula": 0.3, "NAcc": 0.2,
    }
    base.update(overrides)
    return base


def test_clean_run_has_no_flags():
    rep = evaluate(_scores(), "engage")
    assert rep.flags == []
    assert not rep.blocked


def test_high_amygdala_blocks():
    rep = evaluate(_scores(Amygdala=DARK_PATTERN_AMYG + 0.1), "engage")
    assert rep.blocked
    assert any(f.code == "dark_pattern_amygdala" for f in rep.flags)


def test_high_nacc_outside_gamification_blocks():
    rep = evaluate(_scores(NAcc=DARK_PATTERN_NACC + 0.1), "convert")
    assert rep.blocked
    assert any(f.code == "dark_pattern_nacc" for f in rep.flags)


def test_high_nacc_in_gamification_allowed():
    rep = evaluate(_scores(NAcc=DARK_PATTERN_NACC + 0.1), "gamification")
    assert not rep.blocked


def test_yerkes_ceiling_warns():
    rep = evaluate(_scores(V4=AROUSAL_CEILING + 0.05), "engage")
    assert any(f.code == "yerkes_ceiling" for f in rep.flags)


def test_amygdala_regression_warning():
    prev = _scores(Amygdala=0.2)
    cur = _scores(Amygdala=0.35)
    rep = evaluate(cur, "engage", prev_scores=prev)
    assert any(f.code == "amygdala_regression" for f in rep.flags)
