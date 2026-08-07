#!/usr/bin/env python3
"""Unit test for SWA Database Controllers & PGVector RAM Fallback Operations."""

import sys
import os
import numpy as np

# Ensure swa is importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from swa.db_controllers import (
    PostgresBridgeController,
    MemoryPageController,
    CpuAffinityController,
    PCoreSchedulerController,
    EpisodicMemoryController,
    PhaseShifterController,
    FileCrawlerController,
    EventualEmbeddingController
)


def test_postgres_bridge():
    print("🧪 Testing PostgresBridgeController...")
    bridge = PostgresBridgeController(db_uri="")
    assert bridge.is_db_active() is False, "DB should be inactive with empty URI"
    
    # Should not crash and should return False in RAM-only fallback
    success = bridge.verify_schema_and_migrations()
    assert success is False
    print("  PostgresBridgeController RAM fallback OK")


def test_memory_page_controller():
    print("\n🧪 Testing MemoryPageController Page Scans...")
    controller = MemoryPageController(db_uri="")
    
    payload = controller.log_memory_pages()
    assert "total_ram_gb" in payload
    assert "used_ram_gb" in payload
    assert "percent_used" in payload
    assert "semantic_state_signature" in payload
    
    print(f"  RAM Pages Logged: total={payload['total_ram_gb']}GB, used={payload['percent_used']}%")
    print(f"  Semantic Page State Signature: {payload['semantic_state_signature']}")
    print("✅ MemoryPageController verified successfully!")


def test_cpu_affinity_controller():
    print("\n🧪 Testing CpuAffinityController Semantic Bindings...")
    controller = CpuAffinityController(db_uri="")
    
    # Bind affinity to Physical Core 2 with logical CPUs 4 and 5
    payload = controller.save_affinity_state(core_id=2, logical_processors=[4, 5])
    assert payload["core_id"] == 2
    assert payload["logical_processors"] == "4,5"
    assert "semantic_trace" in payload
    assert len(payload["semantic_trace"]) == 2560
    
    # Assert vector norm is 1.0 (L2 Normalized)
    norm = np.linalg.norm(payload["semantic_trace"])
    print(f"  Affinity Trace Vector Norm: {norm:.4f} (Expected: 1.0000)")
    assert np.isclose(norm, 1.0), "Semantic trace vector should be L2 normalized"
    
    print("✅ CpuAffinityController verified successfully!")


def test_pcore_scheduler_controller():
    print("\n🧪 Testing PCoreSchedulerController dynamic telemetries...")
    controller = PCoreSchedulerController(db_uri="")
    
    payload = controller.log_scheduler_state(
        utilization_avg=45.2,
        temp_avg=52.0,
        active_cores=4,
        throttled_cores=0,
        summary="CPU load stable, scheduler optimized"
    )
    assert payload["utilization_average"] == 45.2
    assert payload["temperature_average"] == 52.0
    assert payload["active_cores"] == 4
    assert payload["throttled_cores"] == 0
    assert payload["diagnostic_summary"] == "CPU load stable, scheduler optimized"
    print("✅ PCoreSchedulerController verified successfully!")


def test_episodic_memory_search():
    print("\n🧪 Testing EpisodicMemoryController PGVector Cosine Similarity & Search...")
    controller = EpisodicMemoryController(db_uri="")
    
    # 1. Clear RAM DB anchors
    controller._RAM_DATABASE["swa_episodic_episodes"].clear()
    
    # 2. Insert three distinct concept episodes with 2560d embeddings
    print("  Storing test episodes in the semantic space:")
    # Episode A: Strong positive correlation in initial dimensions
    emb_a = [0.0] * 2560
    emb_a[0] = 1.0
    emb_a[1] = 1.0
    controller.store_episode("Core 0 affinity scheduled successfully", emb_a, "CPU_METRICS", weight=1.5)
    
    # Episode B: Weak negative correlation
    emb_b = [0.0] * 2560
    emb_b[0] = -1.0
    emb_b[1] = -1.0
    controller.store_episode("Severe thermal throttling on physical core 3", emb_b, "CPU_THERMAL", weight=2.0)
    
    # Episode C: Orthogonal vector
    emb_c = [0.0] * 2560
    emb_c[5] = 1.0
    controller.store_episode("RAM buffer pool exhaustion semantic anomaly detected", emb_c, "MEM_SECURITY", weight=1.0)
    
    # 3. Query with a vector similar to Episode A (emb_a)
    query_emb = [0.0] * 2560
    query_emb[0] = 0.8
    query_emb[1] = 0.8
    
    print("  Executing Vector Cosine Similarity Query:")
    results = controller.query_similar_episodes(query_emb, limit=3)
    
    assert len(results) == 3
    
    # Verify rankings based on cosine similarity
    # Episode A should be #1 (similarity ~ 1.0)
    # Episode C should be #2 (similarity ~ 0.0)
    # Episode B should be #3 (similarity ~ -1.0)
    print(f"    Match #1: '{results[0]['content']}' (similarity: {results[0]['similarity']:.4f})")
    print(f"    Match #2: '{results[1]['content']}' (similarity: {results[1]['similarity']:.4f})")
    print(f"    Match #3: '{results[2]['content']}' (similarity: {results[2]['similarity']:.4f})")
    
    assert results[0]["concept_group"] == "CPU_METRICS"
    assert results[0]["similarity"] > 0.99
    
    assert results[1]["concept_group"] == "MEM_SECURITY"
    assert np.isclose(results[1]["similarity"], 0.0)
    
    assert results[2]["concept_group"] == "CPU_THERMAL"
    assert results[2]["similarity"] < -0.99
    
    print("✅ EpisodicMemoryController PGVector & RAM fallbacks verified successfully!")


def test_eventual_embeddings_relation():
    print("\n🧪 Testing EventualEmbeddingController PGVector Relations & Cascades...")
    controller = EventualEmbeddingController(db_uri="")
    
    # 1. Clear RAM eventual DB
    controller._RAM_DATABASE["swa_eventual_embeddings"].clear()
    
    # 2. Insert test dynamic embedding vector mapping
    emb_steer = [0.0] * 2560
    emb_steer[10] = 1.0
    emb_steer[11] = 1.0
    
    # Links state ID 4, CPU Scheduler log 1, RAM log 3
    payload = controller.save_eventual_embedding_relation(
        state_id=4,
        cpu_log_id=1,
        mem_log_id=3,
        eventual_bias_2560=emb_steer,
        phase_shift_angle=0.785  # ~ 45 degrees
    )
    
    assert payload["state_id"] == 4
    assert payload["cpu_log_id"] == 1
    assert payload["mem_log_id"] == 3
    assert payload["phase_shift_angle"] == 0.785
    assert "signature_hash" in payload
    print(f"  Linked dynamic state signature: {payload['signature_hash']}")
    
    # 3. Query dynamic eventual vectors
    query_steer = [0.0] * 2560
    query_steer[10] = 0.9
    query_steer[11] = 0.9
    
    results = controller.query_eventual_embeddings_by_load(query_steer, limit=1)
    assert len(results) == 1
    assert results[0]["state_id"] == 4
    assert results[0]["similarity"] > 0.99
    print(f"  Dynamic semantic similarity check: {results[0]['similarity']:.4f} (Expected: >0.99)")
    
    print("✅ EventualEmbeddingController verified successfully!")


def main():
    print("==================================================")
    print(" SWA Database Controllers Structural & Vector Test")
    print("==================================================")
    
    try:
        test_postgres_bridge()
        test_memory_page_controller()
        test_cpu_affinity_controller()
        test_pcore_scheduler_controller()
        test_episodic_memory_search()
        test_eventual_embeddings_relation()
        
        print("\n🎉 All database and vector controller tests PASSED successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

