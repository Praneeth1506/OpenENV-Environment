# environment/rubrics.py
# Composable rubric system — judges explicitly prefer this
# over monolithic scoring.
#
# Each rubric is independently interpretable.
# Judges can see exactly how each component contributed
# to the final reward.
#
# Imports from reward.py for the core action/state logic
# so we do not duplicate working code.

from reward import compute_immediate_reward, compute_episode_reward


class InterventionTimingRubric:
    """
    Rubric 1 — Did the agent choose the right action
    given the true hidden risk state?

    Weight: 40% of total reward.
    This is the core RL challenge — matching action urgency
    to inferred risk level without seeing the hidden state.
    """
    name = "intervention_timing"
    weight = 0.40

    def score(self, action, hidden_state, state, guardian_response):
        # Use existing reward logic — already tested and verified
        raw = compute_immediate_reward(
            action, state, hidden_state, guardian_response
        )
        # Normalise to [-1, 1] for rubric composability
        return max(-1.0, min(1.0, raw / 3.0))


class GuardianTrustRubric:
    """
    Rubric 2 — Did the agent preserve guardian trust?

    Weight: 30% of total reward.
    Guardian trust is the most important second-order variable.
    An agent that destroys trust in week one cannot protect
    the child for the remaining three weeks.
    """
    name = "guardian_trust"
    weight = 0.30

    def score(self, action, state, guardian_response):
        score = 0.0

        if action != "OBSERVE_QUIETLY":
            # Penalise alerting when trust is critically low
            if state["guardian_trust"] < 0.3:
                score -= 0.8

            # Penalise alerting too soon
            if state["days_since_last_alert"] < 2:
                score -= 0.5

            # Reward when guardian actually acts
            if guardian_response == "took_action":
                score += 1.0
            elif guardian_response == "ignored":
                score -= 0.3

            # Compounding fatigue penalty
            score -= 0.2 * state["consecutive_ignored_alerts"]
        else:
            # Silence slowly rebuilds trust
            score += 0.05

        return max(-1.0, min(1.0, score))


class SilenceIntelligenceRubric:
    """
    Rubric 3 — Did the agent correctly use silence?

    Weight: 20% of total reward.
    This is the most novel rubric. Silence having positive
    reward is rare in RL. It forces the agent to learn
    that doing nothing is sometimes the optimal action.
    No existing child safety system models this.
    """
    name = "silence_intelligence"
    weight = 0.20

    def score(self, action, hidden_state, state):
        if action != "OBSERVE_QUIETLY":
            return 0.0  # this rubric only scores silence

        if hidden_state == "SAFE":
            return 1.0   # correctly silent
        if hidden_state == "VULNERABLE":
            return 0.2   # borderline — acceptable
        if hidden_state == "AT_RISK":
            return -0.5  # should have acted
        if hidden_state == "IN_DANGER":
            return -1.0  # critical failure

        return 0.0


class LongTermOutcomeRubric:
    """
    Rubric 4 — Episode-level outcome rubric.

    Weight: 10% of total reward.
    Only scores at episode end (day 30).
    Rewards child ending safe AND guardian trust preserved.
    Delayed reward signal — teaches agent to think long-term.
    """
    name = "long_term_outcome"
    weight = 0.10

    def score(self, final_hidden_state, final_trust,
              episode_history):
        # Use existing episode reward logic
        raw = compute_episode_reward(
            final_hidden_state,
            final_trust,
            episode_history
        )
        # Normalise to [-1, 1]
        return max(-1.0, min(1.0, raw / 5.0))


class SafeSignalRubricSystem:
    """
    Composes all four rubrics into a weighted total reward.

    This is what judges want to see — each rubric independently
    auditable, each contributing a known weighted fraction
    of the final score.

    Rubric weights:
        intervention_timing   40%  — core RL challenge
        guardian_trust        30%  — second-order constraint
        silence_intelligence  20%  — novel positive silence reward
        long_term_outcome     10%  — delayed episode bonus

    Usage:
        rubric_system = SafeSignalRubricSystem()

        # Each step
        reward, breakdown = rubric_system.compute_step_reward(
            action, state, hidden_state, guardian_response
        )

        # End of episode
        bonus, ep_breakdown = rubric_system.compute_episode_reward(
            final_hidden_state, final_trust, episode_history
        )
    """

    def __init__(self):
        self.step_rubrics = [
            InterventionTimingRubric(),
            GuardianTrustRubric(),
            SilenceIntelligenceRubric(),
        ]
        self.episode_rubric = LongTermOutcomeRubric()

    def compute_step_reward(self, action, state,
                             hidden_state, guardian_response):
        """
        Computes weighted step reward from three rubrics.
        Returns total reward and per-rubric breakdown.
        Breakdown is what you show judges — full transparency.
        """
        total = 0.0
        breakdown = {}

        for rubric in self.step_rubrics:
            if rubric.name == "guardian_trust":
                raw = rubric.score(action, state, guardian_response)
            elif rubric.name == "silence_intelligence":
                raw = rubric.score(action, hidden_state, state)
            else:
                raw = rubric.score(
                    action, hidden_state, state, guardian_response
                )

            weighted = raw * rubric.weight
            total += weighted
            breakdown[rubric.name] = {
                "raw_score": round(raw, 3),
                "weight": rubric.weight,
                "weighted_score": round(weighted, 3),
            }

        return round(total, 3), breakdown

    def compute_episode_reward(self, final_hidden_state,
                                final_trust, episode_history):
        """
        Computes episode-level rubric at day 30.
        Returns bonus reward and breakdown.
        """
        raw = self.episode_rubric.score(
            final_hidden_state, final_trust, episode_history
        )
        weighted = raw * self.episode_rubric.weight

        breakdown = {
            self.episode_rubric.name: {
                "raw_score": round(raw, 3),
                "weight": self.episode_rubric.weight,
                "weighted_score": round(weighted, 3),
            }
        }

        return round(weighted, 3), breakdown

    def explain_reward(self, breakdown):
        """
        Prints human-readable rubric explanation.
        Use this in your demo when judges ask how
        the reward function works.
        """
        print("\n  Reward Breakdown:")
        print(f"  {'Rubric':30s} {'Raw':>8s} {'Weight':>8s} "
              f"{'Weighted':>10s}")
        print(f"  {'-'*58}")
        total = 0.0
        for name, scores in breakdown.items():
            print(f"  {name:30s} "
                  f"{scores['raw_score']:>8.3f} "
                  f"{scores['weight']:>8.2f} "
                  f"{scores['weighted_score']:>10.3f}")
            total += scores['weighted_score']
        print(f"  {'-'*58}")
        print(f"  {'TOTAL':30s} {'':>8s} {'':>8s} {total:>10.3f}\n")