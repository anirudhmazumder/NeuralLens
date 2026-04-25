from neurolens.reward import compute, YERKES_CEILING


def _flat(v: float = 0.5) -> dict[str, float]:
    return {
        "FFA": v, "V4": v, "MT+": v, "Hippocampus": v,
        "PFC": v, "ACC": v, "Amygdala": v, "Insula": v, "NAcc": v,
    }


def test_engage_rewards_face_and_color():
    base = compute(_flat(0.2), "engage")
    boosted = compute({**_flat(0.2), "FFA": 0.9, "V4": 0.9}, "engage")
    assert boosted.total > base.total


def test_trust_prefers_pfc_over_engagement():
    boost_pfc = compute({**_flat(0.2), "PFC": 0.9}, "trust").total
    boost_ffa = compute({**_flat(0.2), "FFA": 0.9}, "trust").total
    assert boost_pfc > boost_ffa


def test_amygdala_is_heavily_penalized():
    calm = compute(_flat(0.2), "engage").total
    anxious = compute({**_flat(0.2), "Amygdala": 0.9}, "engage").total
    assert anxious < calm
    # 1.5 weight => at least ~1.0 drop for 0.7 delta
    assert calm - anxious > 0.8


def test_nacc_is_dual_signal():
    high_nacc = {**_flat(0.2), "NAcc": 0.9}
    as_engage = compute(high_nacc, "engage").total
    as_game = compute(high_nacc, "gamification").total
    assert as_game > as_engage  # gamification rewards NAcc, engage penalizes it


def test_yerkes_ceiling_penalty():
    just_under = compute({**_flat(0.2), "FFA": YERKES_CEILING - 0.01}, "engage")
    over = compute({**_flat(0.2), "FFA": YERKES_CEILING + 0.05}, "engage")
    assert "FFA" in over.yerkes_violations
    assert over.total < just_under.total + 0.1  # ceiling cuts in
