from swa.context_domain import (
    ContextEmbedding,
    ContextIdentity,
    ContextObservation,
    ContextRelation,
    GravityField,
    context_fingerprint,
)


def test_context_identity_is_content_addressed():
    identity = ContextIdentity.from_content("src/a.py", b"print(1)")
    assert len(identity.content_hash) == 64
    assert context_fingerprint(identity) != context_fingerprint(
        ContextIdentity.from_content("src/a.py", b"print(2)")
    )


def test_embedding_and_gravity_field_are_normalized_and_bounded():
    embedding = ContextEmbedding("hash", "test", (3.0, 4.0))
    assert embedding.values == (0.6, 0.8)
    field = GravityField("conversation", ("hash",), (3.0, 4.0), strength=4)
    assert field.objective_vector == (0.6, 0.8)
    assert field.strength == 2.0


def test_observation_and_relation_validate_metrics():
    observation = ContextObservation("hash", "conversation", 0, "meio", 2, -1, "blocked")
    assert observation.semantic_distance == 1.0
    assert observation.convergence == 0.0
    relation = ContextRelation("a", "b", "supports", 2)
    assert relation.similarity == 1.0
