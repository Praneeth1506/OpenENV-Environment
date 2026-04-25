# signals/signal_aggregator.py
# Combines all 7 signal clusters into a unified extended state vector.
# This is what replaces get_observable_state() in simulated_child.py

from signals.reciprocity import ReciprocitySignals
from signals.timing import TimingSignals
from signals.dependency import DependencySignals
from signals.social_graph import SocialGraphSignals
from signals.secrecy import SecrecySignals
from signals.transactions import TransactionSignals
from signals.migration import MigrationSignals

class SignalAggregator:
    """
    Combines all seven behavioral signal clusters into
    a unified state vector for the RL agent.

    Addresses:
    Gap 3 — relationship dynamics (clusters 1,2,4,5,6)
    Gap 6 — cross platform migration (cluster 3)
    Gap 7 — subtle grooming (all clusters)
    """

    def __init__(self, child):
        self.child = child

    def get_full_state(self, base_state, migration_occurred=False,
                       migration_day=None):
        """
        Takes the base state from simulated_child.py and
        extends it with all seven signal clusters.
        """
        risk = self.child.hidden_state_numeric
        archetype = self.child.archetype
        baseline = self.child.baseline
        day = self.child.day

        # Compute all seven signal clusters
        reciprocity = ReciprocitySignals(risk, archetype).compute()
        timing = TimingSignals(risk, archetype, baseline).compute()
        dependency = DependencySignals(risk, archetype).compute()
        social = SocialGraphSignals(risk, archetype, baseline).compute()
        secrecy = SecrecySignals(risk, archetype).compute()
        transactions = TransactionSignals(risk).compute()
        migration = MigrationSignals(
            risk, day, migration_occurred, migration_day
        ).compute()

        # Merge base state with all signal clusters
        extended_state = {
            **base_state,           # existing signals from simulated_child
            **reciprocity,          # cluster 1
            **timing,               # cluster 2
            **migration,            # cluster 3
            **secrecy,              # cluster 4
            **dependency,           # cluster 5
            **social,               # cluster 6
            **transactions,         # cluster 7
        }

        return extended_state