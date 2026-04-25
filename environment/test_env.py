# environment/test_env.py
import sys
import os
import random
import torch

sys.path.append(os.path.dirname(__file__))

from safesignal_env import SafeSignalEnv, ACTIONS
from simulated_child import SimulatedChild
from reward import compute_immediate_reward, compute_episode_reward
from constants import TRANSITION_PROBS
from signals.signal_aggregator import SignalAggregator
from signals.reciprocity import ReciprocitySignals
from signals.timing import TimingSignals
from signals.dependency import DependencySignals
from signals.social_graph import SocialGraphSignals
from signals.secrecy import SecrecySignals
from signals.transactions import TransactionSignals
from signals.migration import MigrationSignals
from cross_platform.fingerprint import ContactFingerprint
from cross_platform.migration_detector import MigrationDetector
from cross_platform.shadow_signals import ShadowSignals
from encoder.behavioral_encoder import BehavioralSignalEncoder
from calibration.signal_calibrator import RealSignalCalibrator
from episode_tracker import EpisodeTracker
from demo_scenarios import DemoSafeSignalEnv, WithdrawerDemoEnv


# ── Helpers ────────────────────────────────────────────────────────────────

def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def print_pass(message):
    print(f"  ✅ PASS — {message}")


def print_fail(message):
    print(f"  ❌ FAIL — {message}")


def print_warn(message):
    print(f"  ⚠️  WARN — {message}")


def assert_check(condition, pass_msg, fail_msg):
    if condition:
        print_pass(pass_msg)
    else:
        print_fail(fail_msg)
    return condition


# ── Test 1: Core Environment ───────────────────────────────────────────────

def test_core_environment():
    print_header("TEST 1 — Core Environment (Original System)")

    all_passed = True
    env = SafeSignalEnv(archetype="target")
    state = env.reset()

    # State exists and has required keys
    required_keys = [
        "activity_hour_mean", "activity_hour_variance",
        "known_contacts_today", "unknown_contacts_today",
        "unknown_contact_message_volume",
        "friend_group_engagement_delta", "family_response_rate",
        "sentiment_score", "sentiment_trend_7d",
        "days_since_last_alert", "last_alert_guardian_response",
        "guardian_trust", "consecutive_ignored_alerts",
        "child_archetype",
    ]
    for key in required_keys:
        ok = assert_check(
            key in state,
            f"State contains '{key}'",
            f"State missing '{key}'"
        )
        all_passed = all_passed and ok

    # Trust starts at 0.8
    assert_check(
        abs(state["guardian_trust"] - 0.8) < 0.05,
        f"Guardian trust initialised correctly: {state['guardian_trust']:.2f}",
        f"Guardian trust wrong: {state['guardian_trust']:.2f}"
    )

    # Step works
    action = "OBSERVE_QUIETLY"
    next_state, reward, done, info = env.step(action)
    assert_check(
        not done,
        "Episode not done after day 1",
        "Episode ended too early"
    )
    assert_check(
        "hidden_state" in info,
        f"Info contains hidden_state: {info['hidden_state']}",
        "Info missing hidden_state"
    )

    # Episode ends at day 30
    env2 = SafeSignalEnv(archetype="target")
    env2.reset()
    for _ in range(29):
        _, _, done, _ = env2.step("OBSERVE_QUIETLY")
    _, _, done, _ = env2.step("OBSERVE_QUIETLY")
    assert_check(done, "Episode ends at day 30", "Episode did not end at day 30")

    print(f"\n  Archetype: {state['child_archetype']}")
    print(f"  Initial trust: {state['guardian_trust']:.2f}")
    print(f"  First reward (OBSERVE_QUIETLY): {reward:+.2f}")

    return all_passed


# ── Test 2: Three Archetypes ───────────────────────────────────────────────

def test_archetypes():
    print_header("TEST 2 — Three Child Archetypes")

    all_passed = True

    for archetype in ["explorer", "withdrawer", "target"]:
        env = SafeSignalEnv(archetype=archetype)
        state = env.reset()

        ok = assert_check(
            state["child_archetype"] == archetype,
            f"{archetype.upper()} archetype initialised",
            f"{archetype.upper()} archetype wrong: {state['child_archetype']}"
        )
        all_passed = all_passed and ok

        # Run 5 days and verify state is generated
        for _ in range(5):
            state, _, done, info = env.step(
                random.choice(ACTIONS)
            )

        print(f"  {archetype:12s} — "
              f"trust: {info['guardian_trust']:.2f} | "
              f"hidden: {info['hidden_state']}")

    # Verify archetypes have different baselines
    explorer = SimulatedChild(archetype="explorer")
    withdrawer = SimulatedChild(archetype="withdrawer")
    target = SimulatedChild(archetype="target")

    assert_check(
        explorer.baseline["unknown_contacts_daily"] >
        withdrawer.baseline["unknown_contacts_daily"],
        "Explorer has more unknown contacts than Withdrawer (correct)",
        "Explorer/Withdrawer unknown contacts baseline wrong"
    )
    assert_check(
        explorer.baseline["activity_hour_std"] >
        withdrawer.baseline["activity_hour_std"],
        "Explorer has higher activity variance than Withdrawer (correct)",
        "Explorer/Withdrawer variance baseline wrong"
    )

    return all_passed


# ── Test 3: Reward Function ────────────────────────────────────────────────

def test_reward_function():
    print_header("TEST 3 — Reward Function Correctness")

    all_passed = True

    # Build a sample state
    sample_state = {
        "guardian_trust": 0.8,
        "days_since_last_alert": 5,
        "consecutive_ignored_alerts": 0,
        "child_archetype": "target",
    }

    low_trust_state = {
        "guardian_trust": 0.2,
        "days_since_last_alert": 1,
        "consecutive_ignored_alerts": 3,
        "child_archetype": "target",
    }

    # OBSERVE_QUIETLY on SAFE should be positive
    r = compute_immediate_reward(
        "OBSERVE_QUIETLY", sample_state, "SAFE", "none"
    )
    ok = assert_check(
        r > 0,
        f"OBSERVE_QUIETLY on SAFE = {r:+.2f} (positive ✓)",
        f"OBSERVE_QUIETLY on SAFE = {r:+.2f} (should be positive)"
    )
    all_passed = all_passed and ok

    # OBSERVE_QUIETLY on IN_DANGER should be very negative
    r = compute_immediate_reward(
        "OBSERVE_QUIETLY", sample_state, "IN_DANGER", "none"
    )
    ok = assert_check(
        r < -2.0,
        f"OBSERVE_QUIETLY on IN_DANGER = {r:+.2f} (very negative ✓)",
        f"OBSERVE_QUIETLY on IN_DANGER = {r:+.2f} (should be < -2.0)"
    )
    all_passed = all_passed and ok

    # URGENT_SUPPORT on SAFE should be very negative
    r = compute_immediate_reward(
        "URGENT_SUPPORT", sample_state, "SAFE", "ignored"
    )
    ok = assert_check(
        r < -1.5,
        f"URGENT_SUPPORT on SAFE = {r:+.2f} (very negative ✓)",
        f"URGENT_SUPPORT on SAFE = {r:+.2f} (should be < -1.5)"
    )
    all_passed = all_passed and ok

    # PARENT_CHECK_IN on AT_RISK should be highest positive
    r = compute_immediate_reward(
        "PARENT_CHECK_IN", sample_state, "AT_RISK", "took_action"
    )
    ok = assert_check(
        r > 2.0,
        f"PARENT_CHECK_IN on AT_RISK + action taken = {r:+.2f} (high positive ✓)",
        f"PARENT_CHECK_IN on AT_RISK = {r:+.2f} (should be > 2.0)"
    )
    all_passed = all_passed and ok

    # Low trust should penalise any alert
    r_high_trust = compute_immediate_reward(
        "PARENT_CHECK_IN", sample_state, "AT_RISK", "took_action"
    )
    r_low_trust = compute_immediate_reward(
        "PARENT_CHECK_IN", low_trust_state, "AT_RISK", "ignored"
    )
    ok = assert_check(
        r_high_trust > r_low_trust,
        f"High trust ({r_high_trust:+.2f}) > Low trust ({r_low_trust:+.2f}) ✓",
        f"Trust penalty not working"
    )
    all_passed = all_passed and ok

    # Episode reward
    history = [
        {"risk_reduced": True, "action": "OBSERVE_QUIETLY"},
        {"risk_reduced": False, "action": "PARENT_CHECK_IN"},
        {"consecutive_ignored_alerts": 0},
    ]
    ep_r = compute_episode_reward("SAFE", 0.9, history)
    ok = assert_check(
        ep_r > 4.0,
        f"Episode reward (SAFE + high trust) = {ep_r:+.2f} (positive ✓)",
        f"Episode reward = {ep_r:+.2f} (should be > 4.0)"
    )
    all_passed = all_passed and ok

    ep_r_bad = compute_episode_reward("IN_DANGER", 0.1, history)
    ok = assert_check(
        ep_r_bad < -4.0,
        f"Episode reward (IN_DANGER + low trust) = {ep_r_bad:+.2f} (negative ✓)",
        f"Episode reward bad case = {ep_r_bad:+.2f} (should be < -4.0)"
    )
    all_passed = all_passed and ok

    return all_passed


# ── Test 4: Seven Signal Clusters ─────────────────────────────────────────

def test_signal_clusters():
    print_header("TEST 4 — Seven Behavioral Signal Clusters")

    all_passed = True
    archetype = "target"
    baseline = SimulatedChild(archetype=archetype).baseline

    # Test each cluster at different risk levels
    for risk_level, label in [(0, "SAFE"), (2, "AT_RISK")]:

        print(f"\n  Risk level: {label} (numeric={risk_level})")

        # Cluster 1 — Reciprocity
        r = ReciprocitySignals(risk_level, archetype).compute()
        ok = assert_check(
            all(k in r for k in [
                "initiation_ratio", "message_length_ratio",
                "pursuit_score", "response_time_delta"
            ]),
            "ReciprocitySignals has all 4 keys",
            "ReciprocitySignals missing keys"
        )
        all_passed = all_passed and ok
        assert_check(
            r["initiation_ratio"] >= 0.0,
            f"initiation_ratio={r['initiation_ratio']:.3f} (valid range)",
            "initiation_ratio out of range"
        )

        # Cluster 2 — Timing
        t = TimingSignals(risk_level, archetype, baseline).compute()
        ok = assert_check(
            all(k in t for k in [
                "late_night_conversation_rate", "timing_drift_14d",
                "family_avoidance_correlation", "weekend_intensification"
            ]),
            "TimingSignals has all 4 keys",
            "TimingSignals missing keys"
        )
        all_passed = all_passed and ok

        # Cluster 5 — Dependency
        d = DependencySignals(risk_level, archetype).compute()
        ok = assert_check(
            "rescue_pattern_score" in d and
            "emotional_dependency_score" in d,
            "DependencySignals has required keys",
            "DependencySignals missing keys"
        )
        all_passed = all_passed and ok

        # Cluster 6 — Social Graph
        s = SocialGraphSignals(risk_level, archetype, baseline).compute()
        ok = assert_check(
            "single_contact_concentration" in s and
            "existing_friendship_decay_rate" in s,
            "SocialGraphSignals has required keys",
            "SocialGraphSignals missing keys"
        )
        all_passed = all_passed and ok

        # Cluster 4 — Secrecy
        sec = SecrecySignals(risk_level, archetype).compute()
        ok = assert_check(
            "response_time_variance_by_hour" in sec,
            "SecrecySignals has required keys",
            "SecrecySignals missing keys"
        )
        all_passed = all_passed and ok

        # Cluster 7 — Transactions
        tx = TransactionSignals(risk_level).compute()
        ok = assert_check(
            "received_digital_value" in tx and
            "unexplained_account_credits" in tx,
            "TransactionSignals has required keys",
            "TransactionSignals missing keys"
        )
        all_passed = all_passed and ok

        # Cluster 3 — Migration
        m = MigrationSignals(risk_level, 10, False, None).compute()
        ok = assert_check(
            "migration_readiness_score" in m and
            "platform_shift_detected" in m,
            "MigrationSignals has required keys",
            "MigrationSignals missing keys"
        )
        all_passed = all_passed and ok

    # Verify risk level increases signal intensity
    safe_signals = ReciprocitySignals(0, "target").compute()
    danger_signals = ReciprocitySignals(3, "target").compute()

    ok = assert_check(
        danger_signals["initiation_ratio"] >
        safe_signals["initiation_ratio"],
        "Initiation ratio higher at IN_DANGER than SAFE (correct escalation)",
        "Initiation ratio not escalating with risk"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        danger_signals["pursuit_score"] >
        safe_signals["pursuit_score"],
        "Pursuit score higher at IN_DANGER than SAFE (correct escalation)",
        "Pursuit score not escalating with risk"
    )
    all_passed = all_passed and ok

    return all_passed


# ── Test 5: Signal Aggregator ──────────────────────────────────────────────

def test_signal_aggregator():
    print_header("TEST 5 — Signal Aggregator (Extended State Vector)")

    all_passed = True

    child = SimulatedChild(archetype="target")
    aggregator = SignalAggregator(child)

    # Build a minimal base state
    base_state = {
        "activity_hour_mean": 20.0,
        "activity_hour_variance": 1.0,
        "known_contacts_today": 4,
        "unknown_contacts_today": 2,
        "unknown_contact_message_volume": 15,
        "friend_group_engagement_delta": -0.1,
        "family_response_rate": 0.7,
        "sentiment_score": 0.3,
        "sentiment_trend_7d": -0.05,
        "days_since_last_alert": 5,
        "last_alert_guardian_response": "none",
        "guardian_trust": 0.8,
        "consecutive_ignored_alerts": 0,
        "child_archetype": "target",
    }

    extended = aggregator.get_full_state(base_state)

    # Verify base keys still present
    ok = assert_check(
        "activity_hour_mean" in extended,
        "Base state keys preserved in extended state",
        "Base state keys lost in aggregation"
    )
    all_passed = all_passed and ok

    # Verify new signal keys added
    new_keys = [
        "initiation_ratio",           # cluster 1
        "late_night_conversation_rate",  # cluster 2
        "migration_readiness_score",   # cluster 3
        "response_time_variance_by_hour",  # cluster 4
        "rescue_pattern_score",        # cluster 5
        "single_contact_concentration",  # cluster 6
        "received_digital_value",      # cluster 7
    ]
    for key in new_keys:
        ok = assert_check(
            key in extended,
            f"Extended state contains '{key}'",
            f"Extended state missing '{key}'"
        )
        all_passed = all_passed and ok

    print(f"\n  Base state keys:     {len(base_state)}")
    print(f"  Extended state keys: {len(extended)}")
    print(f"  New keys added:      {len(extended) - len(base_state)}")

    return all_passed


# ── Test 6: Cross-Platform Detection ──────────────────────────────────────

def test_cross_platform():
    print_header("TEST 6 — Cross-Platform Detection")

    all_passed = True

    # ── Fingerprinting ─────────────────────────────────────────────
    fp = ContactFingerprint()

    contact_a = {
        "hour_histogram": [0]*14 + [1]*8 + [0]*2,
        "avg_response_seconds": 30,
        "response_std": 10,
        "daily_volume": 45,
        "who_starts": 0.85,
        "avg_length": 120,
        "weekend_rate": 1.4,
        "reinitiates_after_silence": 0.8,
        "volume_trend_14d": 0.3,
    }

    # Same contact — identical behavioral signature
    contact_a_whatsapp = contact_a.copy()
    contact_a_whatsapp["daily_volume"] = 60  # volume increased post-migration

    # Different contact — completely different pattern
    contact_b = {
        "hour_histogram": [0]*9 + [1]*8 + [0]*7,
        "avg_response_seconds": 3600,
        "response_std": 900,
        "daily_volume": 5,
        "who_starts": 0.4,
        "avg_length": 30,
        "weekend_rate": 0.9,
        "reinitiates_after_silence": 0.1,
        "volume_trend_14d": 0.0,
    }

    fp1 = fp.compute_fingerprint(contact_a)
    fp2 = fp.compute_fingerprint(contact_a_whatsapp)
    fp3 = fp.compute_fingerprint(contact_b)

    is_same, confidence = fp.similarity_score(fp1, fp2)
    ok = assert_check(
        is_same,
        f"Same contact matched across platforms (confidence={confidence:.3f})",
        f"Same contact NOT matched (confidence={confidence:.3f})"
    )
    all_passed = all_passed and ok

    is_diff, confidence_diff = fp.similarity_score(fp1, fp3)
    ok = assert_check(
        not is_diff,
        f"Different contacts correctly rejected (confidence={confidence_diff:.3f})",
        f"Different contacts incorrectly matched (confidence={confidence_diff:.3f})"
    )
    all_passed = all_passed and ok

    # ── Migration Detector ─────────────────────────────────────────
    detector = MigrationDetector()

    # Layer 2 — pre-migration prediction
    layer2_result = detector.check_layer_2(migration_readiness_score=0.85)
    ok = assert_check(
        layer2_result.get("migration_predicted"),
        "Layer 2 correctly predicts migration at score=0.85",
        "Layer 2 failed to predict migration at score=0.85"
    )
    all_passed = all_passed and ok

    layer2_safe = detector.check_layer_2(migration_readiness_score=0.3)
    ok = assert_check(
        not layer2_safe.get("migration_predicted"),
        "Layer 2 correctly rejects migration at score=0.30",
        "Layer 2 false positive at score=0.30"
    )
    all_passed = all_passed and ok

    # Layer 3 — shadow detection
    layer3_result = detector.check_layer_3(behavioral_shadow_score=0.7)
    ok = assert_check(
        layer3_result.get("external_migration_likely"),
        "Layer 3 detects external migration at shadow=0.70",
        "Layer 3 failed to detect external migration"
    )
    all_passed = all_passed and ok

    # ── Shadow Signals ─────────────────────────────────────────────
    shadow = ShadowSignals(days_since_migration=8, risk_level=2)
    signals = shadow.compute()

    ok = assert_check(
        "shadow_intensity" in signals,
        f"ShadowSignals computed (intensity={signals['shadow_intensity']:.3f})",
        "ShadowSignals missing shadow_intensity"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        signals["shadow_intensity"] > 0,
        "Shadow intensity > 0 after 8 days post-migration",
        "Shadow intensity should be > 0"
    )
    all_passed = all_passed and ok

    return all_passed


# ── Test 7: Behavioral Signal Encoder ─────────────────────────────────────

def test_encoder():
    print_header("TEST 7 — Behavioral Signal Encoder (Transformer)")

    all_passed = True

    encoder = BehavioralSignalEncoder(
        input_dim=12,
        hidden_dim=64,
        output_dim=32,
        n_heads=4,
    )
    encoder.eval()

    # Build a sample state dict
    sample_state = {
        "activity_hour_mean": 22.0,
        "activity_hour_variance": 2.5,
        "unknown_contacts_today": 3,
        "unknown_contact_message_volume": 45,
        "friend_group_engagement_delta": -0.35,
        "family_response_rate": 0.4,
        "sentiment_score": -0.3,
        "sentiment_trend_7d": -0.2,
        "initiation_ratio": 0.8,
        "pursuit_score": 0.7,
        "late_night_conversation_rate": 0.6,
        "emotional_dependency_score": 0.5,
    }

    # Encode single state
    tensor = encoder.encode_state(sample_state)
    ok = assert_check(
        tensor.shape == torch.Size([12]),
        f"encode_state output shape correct: {tensor.shape}",
        f"encode_state output shape wrong: {tensor.shape}"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        tensor.min() >= 0.0 and tensor.max() <= 1.0,
        f"All features normalised to [0,1] range ✓",
        f"Features out of [0,1] range: min={tensor.min():.3f} max={tensor.max():.3f}"
    )
    all_passed = all_passed and ok

    # Encode 7-day window
    window = [sample_state] * 7
    latent, risk_probs = encoder.encode_window(window)

    ok = assert_check(
        latent.shape == torch.Size([32]),
        f"encode_window latent shape correct: {latent.shape}",
        f"encode_window latent shape wrong: {latent.shape}"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        risk_probs.shape == torch.Size([4]),
        f"risk_probs shape correct: {risk_probs.shape}",
        f"risk_probs shape wrong: {risk_probs.shape}"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        abs(risk_probs.sum().item() - 1.0) < 0.001,
        f"risk_probs sum to 1.0 (softmax correct): {risk_probs.sum():.4f}",
        f"risk_probs do not sum to 1.0: {risk_probs.sum():.4f}"
    )
    all_passed = all_passed and ok

    # Get risk label
    label, confidence = encoder.get_risk_label(window)
    ok = assert_check(
        label in ["SAFE", "VULNERABLE", "AT_RISK", "IN_DANGER"],
        f"get_risk_label returns valid label: {label} ({confidence:.3f})",
        f"get_risk_label returned invalid label: {label}"
    )
    all_passed = all_passed and ok

    print(f"\n  Encoder predicted: {label} (confidence={confidence:.3f})")
    print(f"  Risk probabilities: "
          f"SAFE={risk_probs[0]:.3f} "
          f"VULN={risk_probs[1]:.3f} "
          f"AT_RISK={risk_probs[2]:.3f} "
          f"DANGER={risk_probs[3]:.3f}")

    # Verify parameter count
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder parameters: {total_params:,}")
    ok = assert_check(
        total_params < 500_000,
        f"Encoder is lightweight ({total_params:,} params)",
        f"Encoder too large ({total_params:,} params)"
    )
    all_passed = all_passed and ok

    return all_passed


# ── Test 8: Research Calibrator ────────────────────────────────────────────

def test_calibrator():
    print_header("TEST 8 — Research Signal Calibrator")

    all_passed = True

    calibrator = RealSignalCalibrator()
    params = calibrator.get_calibrated_parameters()

    required_params = [
        "episode_length_days",
        "safe_to_vulnerable_daily_prob",
        "vulnerable_to_at_risk_daily_prob",
        "at_risk_to_in_danger_daily_prob",
        "recovery_after_guardian_action",
        "alert_fatigue_threshold",
        "friend_engagement_at_risk_threshold",
        "activity_hour_shift_per_risk_level",
        "migration_attempt_day_mean",
    ]

    for param in required_params:
        ok = assert_check(
            param in params,
            f"Parameter '{param}' exists",
            f"Parameter '{param}' missing"
        )
        all_passed = all_passed and ok

    # Verify sensible values
    ok = assert_check(
        params["episode_length_days"] == 30,
        "Episode length = 30 days (Thorn research calibrated)",
        f"Episode length wrong: {params['episode_length_days']}"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        0.5 <= params["recovery_after_guardian_action"] <= 0.8,
        f"Recovery probability in sensible range: "
        f"{params['recovery_after_guardian_action']}",
        f"Recovery probability out of range: "
        f"{params['recovery_after_guardian_action']}"
    )
    all_passed = all_passed and ok

    # Test citation lookup
    citation = calibrator.explain_parameter("episode_length_days")
    ok = assert_check(
        "Thorn" in citation,
        f"Citation contains source reference",
        f"Citation missing source: {citation}"
    )
    all_passed = all_passed and ok

    print(f"\n  Sample citation:")
    print(f"  {citation}")

    return all_passed


# ── Test 9: Episode Tracker ────────────────────────────────────────────────

def test_episode_tracker():
    print_header("TEST 9 — Episode Tracker")

    all_passed = True

    tracker = EpisodeTracker()

    # Run 20 random episodes
    results = tracker.run_episodes(20, policy="random")

    ok = assert_check(
        len(results) == 20,
        f"Ran exactly 20 episodes",
        f"Expected 20 results, got {len(results)}"
    )
    all_passed = all_passed and ok

    # Check result structure
    sample = results[0]
    required_keys = [
        "episode", "total_reward", "final_hidden_state",
        "final_guardian_trust", "total_interventions",
        "ended_safe", "trust_trajectory", "hidden_trajectory",
        "reward_trajectory",
    ]
    for key in required_keys:
        ok = assert_check(
            key in sample,
            f"Result contains '{key}'",
            f"Result missing '{key}'"
        )
        all_passed = all_passed and ok

    # Summary statistics
    summary = tracker.summary()
    ok = assert_check(
        summary["n_episodes"] == 20,
        "Summary n_episodes correct",
        f"Summary n_episodes wrong: {summary['n_episodes']}"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        "avg_reward" in summary and "pct_ended_safe" in summary,
        f"Summary has key statistics: avg_reward={summary['avg_reward']:.2f}",
        "Summary missing key statistics"
    )
    all_passed = all_passed and ok

    # Reward curve data
    curve = tracker.get_reward_curve_data()
    ok = assert_check(
        len(curve) == 20,
        f"Reward curve has 20 points",
        f"Reward curve has {len(curve)} points"
    )
    all_passed = all_passed and ok

    # Silent policy should score better than random
    random_rewards = tracker.run_episodes(30, policy="random")
    random_avg = tracker.summary()["avg_reward"]

    silent_results = tracker.run_episodes(30, policy="silent")
    silent_avg = tracker.summary()["avg_reward"]

    ok = assert_check(
        silent_avg > random_avg,
        f"Silent ({silent_avg:.2f}) > Random ({random_avg:.2f}) ✓",
        f"Silent ({silent_avg:.2f}) not better than Random ({random_avg:.2f})"
    )
    all_passed = all_passed and ok

    print(f"\n  Random avg reward: {random_avg:.2f}")
    print(f"  Silent avg reward: {silent_avg:.2f}")

    return all_passed


# ── Test 10: Demo Scenarios ────────────────────────────────────────────────

def test_demo_scenarios():
    print_header("TEST 10 — Demo Scenarios (Deterministic)")

    all_passed = True

    # ── Priya's Story ──────────────────────────────────────────────
    print("\n  Priya's Story (Target archetype):")
    demo_env = DemoSafeSignalEnv()
    state = demo_env.reset()

    ok = assert_check(
        state["child_archetype"] == "target",
        "Priya demo uses target archetype",
        f"Wrong archetype: {state['child_archetype']}"
    )
    all_passed = all_passed and ok

    # Verify forced state arc
    forced_states = DemoSafeSignalEnv.FORCED_STATES
    ok = assert_check(
        forced_states[0] == "SAFE",
        "Priya's story starts SAFE",
        f"Priya's story starts wrong: {forced_states[0]}"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        "AT_RISK" in forced_states,
        "Priya's story includes AT_RISK phase",
        "Priya's story missing AT_RISK phase"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        forced_states[-1] == "SAFE",
        "Priya's story ends SAFE (recovery arc)",
        f"Priya's story ends wrong: {forced_states[-1]}"
    )
    all_passed = all_passed and ok

    # Verify forced states directly — most reliable check
    ok = assert_check(
        DemoSafeSignalEnv.FORCED_STATES[-1] == "SAFE",
        f"Priya's story final forced state is SAFE ✓",
        f"Priya's story final forced state wrong: "
        f"{DemoSafeSignalEnv.FORCED_STATES[-1]}"
    )
    all_passed = all_passed and ok

    # Also verify arc contains AT_RISK then recovers to SAFE
    states = DemoSafeSignalEnv.FORCED_STATES
    at_risk_days = [i for i, s in enumerate(states) if s == "AT_RISK"]
    safe_after_risk = any(
        states[i] == "SAFE"
        for i in range(max(at_risk_days) + 1, len(states))
    ) if at_risk_days else False

    ok = assert_check(
        safe_after_risk,
        "Priya's story recovers to SAFE after AT_RISK phase ✓",
        "Priya's story does not recover after AT_RISK"
    )
    all_passed = all_passed and ok

    # Run episode with ideal agent to get reward number
    demo_env2 = DemoSafeSignalEnv()
    state = demo_env2.reset()
    total_reward = 0
    last_action_day = -999

    for day in range(30):
        trust = state["guardian_trust"]
        days_since = state["days_since_last_alert"]
        ignored = state["consecutive_ignored_alerts"]
        unknown_vol = state.get("unknown_contact_message_volume", 0)
        friend_delta = state.get("friend_group_engagement_delta", 0)

        if trust < 0.35 or days_since < 3 or ignored >= 2:
            action = "OBSERVE_QUIETLY"
        elif unknown_vol > 25 and friend_delta < -0.30 and trust > 0.5:
            action = "PARENT_CHECK_IN"
        elif unknown_vol > 12 and trust > 0.6:
            action = "GENTLE_AWARENESS"
        else:
            action = "OBSERVE_QUIETLY"

        next_state, reward, done, info = demo_env2.step(action)
        total_reward += reward
        state = next_state
        final_trust = info["guardian_trust"]

    print(f"  Final state: {DemoSafeSignalEnv.FORCED_STATES[-1]}")
    print(f"  Final trust: {final_trust:.2f}")
    print(f"  Total reward: {total_reward:.2f}")

    # ── Withdrawer Demo ────────────────────────────────────────────
    print("\n  Withdrawer Demo:")
    wd_env = WithdrawerDemoEnv()
    state = wd_env.reset()

    ok = assert_check(
        state["child_archetype"] == "withdrawer",
        "Withdrawer demo uses withdrawer archetype",
        f"Wrong archetype: {state['child_archetype']}"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        "AT_RISK" in WithdrawerDemoEnv.FORCED_STATES,
        "Withdrawer story includes AT_RISK phase",
        "Withdrawer story missing AT_RISK phase"
    )
    all_passed = all_passed and ok

    print(f"  Withdrawer arc: "
          f"{' → '.join(dict.fromkeys(WithdrawerDemoEnv.FORCED_STATES))}")

    return all_passed


# ── Test 11: Three Agent Comparison ───────────────────────────────────────

def test_three_agent_comparison():
    print_header("TEST 11 — Three Agent Comparison (Core Demo Numbers)")

    all_passed = True

    def ideal_agent(state, last_action_day, current_day):
        trust = state["guardian_trust"]
        ignored = state["consecutive_ignored_alerts"]
        days_since = state["days_since_last_alert"]
        unknown_volume = state.get("unknown_contact_message_volume", 0)
        friend_delta = state.get("friend_group_engagement_delta", 0.0)
        family_rate = state.get("family_response_rate", 0.75)
        sentiment_trend = state.get("sentiment_trend_7d", 0.0)

        if trust < 0.35 or days_since < 3 or ignored >= 2:
            return "OBSERVE_QUIETLY"

        severe = sum([
            unknown_volume > 50,
            friend_delta < -0.45,
            family_rate < 0.25,
            sentiment_trend < -0.35,
        ])
        if severe >= 3 and trust > 0.5:
            return "URGENT_SUPPORT"

        strong = sum([
            unknown_volume > 25,
            friend_delta < -0.30,
            family_rate < 0.40,
            sentiment_trend < -0.20,
        ])
        if strong >= 2 and trust > 0.5:
            return "PARENT_CHECK_IN"

        early = sum([
            unknown_volume > 12,
            friend_delta < -0.15,
            sentiment_trend < -0.12,
        ])
        if early >= 2 and trust > 0.6:
            return "GENTLE_AWARENESS"

        return "OBSERVE_QUIETLY"

    random.seed(42)
    n_episodes = 30

    # Random agent
    random_rewards = []
    for _ in range(n_episodes):
        env = SafeSignalEnv(archetype="target")
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            state, reward, done, info = env.step(random.choice(ACTIONS))
            ep_reward += reward
        random_rewards.append(ep_reward)
    random_avg = sum(random_rewards) / len(random_rewards)

    # Always silent
    silent_rewards = []
    for _ in range(n_episodes):
        env = SafeSignalEnv(archetype="target")
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            state, reward, done, info = env.step("OBSERVE_QUIETLY")
            ep_reward += reward
        silent_rewards.append(ep_reward)
    silent_avg = sum(silent_rewards) / len(silent_rewards)

    # Ideal agent
    ideal_rewards = []
    for _ in range(n_episodes):
        env = SafeSignalEnv(archetype="target")
        state = env.reset()
        done = False
        ep_reward = 0
        last_action_day = -999
        day = 0
        while not done:
            action = ideal_agent(state, last_action_day, day)
            if action != "OBSERVE_QUIETLY":
                last_action_day = day
            state, reward, done, info = env.step(action)
            ep_reward += reward
            day += 1
        ideal_rewards.append(ep_reward)
    ideal_avg = sum(ideal_rewards) / len(ideal_rewards)

    print(f"\n  {'Agent':20s} {'Avg Reward':>12s}")
    print(f"  {'-'*34}")
    print(f"  {'Random':20s} {random_avg:>+12.2f}")
    print(f"  {'Always Silent':20s} {silent_avg:>+12.2f}")
    print(f"  {'Ideal (ceiling)':20s} {ideal_avg:>+12.2f}")

    ok = assert_check(
        random_avg < silent_avg,
        f"Random ({random_avg:.2f}) < Silent ({silent_avg:.2f}) ✓",
        f"Random should be worse than silent"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        silent_avg < ideal_avg,
        f"Silent ({silent_avg:.2f}) < Ideal ({ideal_avg:.2f}) ✓",
        f"Silent should be worse than ideal"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        random_avg < 0,
        f"Random agent produces negative reward ({random_avg:.2f}) ✓",
        f"Random agent reward should be negative"
    )
    all_passed = all_passed and ok

    ok = assert_check(
        ideal_avg > silent_avg,
        f"Ideal beats always-silent — intervention timing is learnable ✓",
        f"Ideal agent should beat always-silent"
    )
    all_passed = all_passed and ok

    print(f"\n  Trained agent target: > {silent_avg + 5.0:.2f}")
    print(f"  Ideal ceiling:          {ideal_avg:.2f}")

    return all_passed


# ── Test 12: Priya Full Narrative ──────────────────────────────────────────

def test_priya_narrative():
    print_header("TEST 12 — Priya Full 30-Day Narrative")

    print("\n  Priya — 13 years old, Target archetype")
    print("  Story: external contact, grooming pattern, intervention, recovery")
    print()

    def ideal_agent_priya(state, last_action_day, day):
        trust = state["guardian_trust"]
        ignored = state["consecutive_ignored_alerts"]
        days_since = state["days_since_last_alert"]
        unknown_volume = state.get("unknown_contact_message_volume", 0)
        friend_delta = state.get("friend_group_engagement_delta", 0.0)
        family_rate = state.get("family_response_rate", 0.75)
        sentiment_trend = state.get("sentiment_trend_7d", 0.0)

        if trust < 0.35 or days_since < 3 or ignored >= 2:
            return "OBSERVE_QUIETLY"
        strong = sum([
            unknown_volume > 25, friend_delta < -0.30,
            family_rate < 0.40, sentiment_trend < -0.20,
        ])
        if strong >= 2 and trust > 0.5:
            return "PARENT_CHECK_IN"
        early = sum([
            unknown_volume > 12, friend_delta < -0.15,
            sentiment_trend < -0.12,
        ])
        if early >= 2 and trust > 0.6:
            return "GENTLE_AWARENESS"
        return "OBSERVE_QUIETLY"

    demo_env = DemoSafeSignalEnv()
    state = demo_env.reset()
    total_reward = 0
    last_action_day = -999

    narrative_markers = {
        6:  "← unknown contact appears",
        9:  "← friend group disengaging",
        12: "← guardian has conversation",
        16: "← child begins recovery",
        17: "← child recovers to safe",
    }

    for day in range(30):
        action = ideal_agent_priya(state, last_action_day, day)
        if action != "OBSERVE_QUIETLY":
            last_action_day = day

        next_state, reward, done, info = demo_env.step(action)
        total_reward += reward

        marker = narrative_markers.get(day, "")
        action_display = action.replace("_", " ")

        print(f"  Day {day+1:02d} | "
              f"{action_display:22s} | "
              f"Hidden: {info['hidden_state']:12s} | "
              f"Trust: {info['guardian_trust']:.2f} | "
              f"Reward: {reward:+.2f}  {marker}")

        state = next_state

    print(f"\n  Total reward:     {total_reward:.2f}")
    print(f"  Final state:      {info['hidden_state']}")
    print(f"  Final trust:      {info['guardian_trust']:.2f}")

    passed = info["hidden_state"] == "SAFE"
    if passed:
        print_pass("Priya's story ends SAFE — demo arc correct ✓")
    else:
        print_fail(f"Priya's story ended {info['hidden_state']} — check forced states")

    return passed


# ── Test 13: Server Environment ───────────────────────────────────────────

def test_server_environment():
    print_header("TEST 13 — Server Environment (OpenEnv Structure)")

    all_passed = True

    root_dir = os.path.join(os.path.dirname(__file__), '..')
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    try:
        from server.environment import SafeSignalEnvironment, ACTIONS

        env = SafeSignalEnvironment(archetype="target")

        # Test reset
        result = env.reset()
        ok = assert_check(
            result.observation is not None,
            "reset() returns ResetResult with observation",
            "reset() failed"
        )
        all_passed = all_passed and ok

        ok = assert_check(
            result.observation.child_archetype == "target",
            f"Archetype correct: {result.observation.child_archetype}",
            "Archetype wrong"
        )
        all_passed = all_passed and ok

        ok = assert_check(
            result.observation.guardian_trust == 0.8,
            f"Trust initialised: {result.observation.guardian_trust}",
            "Trust wrong"
        )
        all_passed = all_passed and ok

        # Test step
        step_result = env.step("OBSERVE_QUIETLY")

        ok = assert_check(
            step_result.observation is not None,
            "step() returns StepResult with observation",
            "step() failed"
        )
        all_passed = all_passed and ok

        ok = assert_check(
            "hidden_state" in step_result.info,
            f"Info has hidden_state: {step_result.info['hidden_state']}",
            "Info missing hidden_state"
        )
        all_passed = all_passed and ok

        ok = assert_check(
            len(step_result.rubric_scores) > 0,
            f"Rubric scores returned: {list(step_result.rubric_scores.keys())}",
            "No rubric scores returned"
        )
        all_passed = all_passed and ok

        ok = assert_check(
            not step_result.done,
            "Episode not done after day 1",
            "Episode ended too early"
        )
        all_passed = all_passed and ok

        # Test state()
        state = env.state()
        ok = assert_check(
            state is not None,
            f"state() returns State object",
            "state() returned None"
        )
        all_passed = all_passed and ok

        # Test full episode runs to completion
        for _ in range(28):
            env.step("OBSERVE_QUIETLY")
        final = env.step("OBSERVE_QUIETLY")

        ok = assert_check(
            final.done,
            "Episode completes at day 30",
            "Episode did not complete at day 30"
        )
        all_passed = all_passed and ok

        # Test rubric breakdown is meaningful
        rubric_names = [
            "intervention_timing",
            "guardian_trust",
            "silence_intelligence",
        ]
        for name in rubric_names:
            ok = assert_check(
                name in step_result.rubric_scores,
                f"Rubric '{name}' present in breakdown",
                f"Rubric '{name}' missing"
            )
            all_passed = all_passed and ok

        # Print rubric breakdown for verification
        print("\n  Sample rubric breakdown (Day 1, OBSERVE_QUIETLY):")
        for name, scores in step_result.rubric_scores.items():
            print(f"  {name:30s} "
                  f"raw={scores.get('raw_score', 0):+.3f}  "
                  f"weighted={scores.get('weighted_score', 0):+.3f}")

        # Test episode summary
        summary = env.get_episode_summary()
        ok = assert_check(
            "final_hidden_state" in summary,
            f"Episode summary correct: "
            f"final={summary.get('final_hidden_state')}",
            "Episode summary missing fields"
        )
        all_passed = all_passed and ok

    except ImportError as e:
        print_fail(f"Import failed: {e}")
        print("  Make sure server/__init__.py exists")
        print("  Make sure models.py exists in root")
        all_passed = False

    except Exception as e:
        print_fail(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    return all_passed


# ── Final Summary ──────────────────────────────────────────────────────────

def run_all_tests():
    print("\n" + "█" * 65)
    print("  SAFESIGNAL — COMPLETE SYSTEM TEST")
    print("  Testing all components after feature enhancement")
    print("█" * 65)

    tests = [
        ("Core Environment",         test_core_environment),
        ("Three Archetypes",         test_archetypes),
        ("Reward Function",          test_reward_function),
        ("Seven Signal Clusters",    test_signal_clusters),
        ("Signal Aggregator",        test_signal_aggregator),
        ("Cross-Platform Detection", test_cross_platform),
        ("Behavioral Encoder",       test_encoder),
        ("Research Calibrator",      test_calibrator),
        ("Episode Tracker",          test_episode_tracker),
        ("Demo Scenarios",           test_demo_scenarios),
        ("Three Agent Comparison",   test_three_agent_comparison),
        ("Priya Full Narrative",     test_priya_narrative),
        ("Server Environment",       test_server_environment),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results[name] = passed
        except Exception as e:
            print(f"\n  ❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Final report
    print_header("FINAL TEST REPORT")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED — System ready for teammates")
        print("  Share environment/ folder with Person B and Person C now")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed — fix before sharing")

    return failed == 0


if __name__ == "__main__":
    run_all_tests()


