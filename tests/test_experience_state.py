from swa.experience_state import ExperienceState


def test_experience_properties_are_bounded_and_derived():
    state = ExperienceState(
        conversation_key="c",
        wave_index=0,
        information_absorption=1.4,
        semantic_change=0.8,
        salience=0.5,
        coherence=0.6,
    )
    assert state.integration_score == 0.6
    assert state.meaningful_change == 0.4
    assert state.metrics["information_absorption"] == 1.0
    assert state.metrics["coherence"] == 0.6
