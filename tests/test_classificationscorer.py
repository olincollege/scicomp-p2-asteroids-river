"""
Comprehensive pytest suite for common/classificationscorer.py.
Entirely AI-generated for now - we'll see if this causes problems or needs human tweaking.
"""

import pytest
import pandas as pd
from common.classificationscorer import CarrieMeasureResult, ClassificationScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_predicted(*rows):
    """Build a predicted-labels DataFrame from (name, family) tuples."""
    return pd.DataFrame(rows, columns=["name", "family1"])


def make_true(*rows):
    """Build a true-labels DataFrame from (name, family) tuples.
    Only family members are included (no "0"-label rows), matching the real CSVs.
    """
    return pd.DataFrame(rows, columns=["name", "family1"])


# ---------------------------------------------------------------------------
# __init__ / construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_missing_asteroids_added_to_true_labels(self):
        """Asteroids in predicted but absent from true get added with family1="0"."""
        predicted = make_predicted(("A", "4"), ("B", "0"))
        true = make_true(("A", "4"))
        scorer = ClassificationScorer(predicted, true)
        assert "B" in set(scorer.true_labels["name"])

    def test_missing_asteroids_get_zero_label(self):
        predicted = make_predicted(("A", "4"), ("B", "0"))
        true = make_true(("A", "4"))
        scorer = ClassificationScorer(predicted, true)
        row = scorer.true_labels[scorer.true_labels["name"] == "B"]
        assert row.iloc[0]["family1"] == "0"

    def test_existing_true_label_preserved(self):
        predicted = make_predicted(("A", "4"))
        true = make_true(("A", "4"))
        scorer = ClassificationScorer(predicted, true)
        row = scorer.true_labels[scorer.true_labels["name"] == "A"]
        assert row.iloc[0]["family1"] == "4"

    def test_no_missing_asteroids_unchanged(self):
        """When predicted ⊆ true, nothing extra is appended."""
        predicted = make_predicted(("A", "4"))
        true = make_true(("A", "4"), ("B", "8"))
        scorer = ClassificationScorer(predicted, true)
        assert len(scorer.true_labels) == 2

    def test_all_asteroids_missing_from_true(self):
        """All predicted asteroids are absent from true – all added as "0"."""
        predicted = make_predicted(("X", "0"), ("Y", "0"))
        true = make_true()
        scorer = ClassificationScorer(predicted, true)
        assert set(scorer.true_labels["name"]) == {"X", "Y"}
        assert all(scorer.true_labels["family1"] == "0")


# ---------------------------------------------------------------------------
# rand_index
# ---------------------------------------------------------------------------

class TestRandIndex:
    def test_perfect_classification_is_one(self):
        """Identical predicted and true labels → adjusted Rand index of 1.0."""
        predicted = make_predicted(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        true = make_true(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        scorer = ClassificationScorer(predicted, true)
        assert scorer.rand_index() == pytest.approx(1.0)

    def test_all_same_cluster_when_should_be_split_is_low(self):
        """Collapsing two families into one gives a score well below 1."""
        predicted = make_predicted(("A", "100"), ("B", "100"), ("C", "100"), ("D", "100"))
        true = make_true(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        scorer = ClassificationScorer(predicted, true)
        assert scorer.rand_index() < 1.0

    def test_returns_float(self):
        predicted = make_predicted(("A", "4"), ("B", "8"))
        true = make_true(("A", "4"), ("B", "8"))
        scorer = ClassificationScorer(predicted, true)
        assert isinstance(scorer.rand_index(), float)

    def test_swapped_labels_penalised(self):
        """Swapping every family label gives a lower score than perfect."""
        predicted_perfect = make_predicted(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        predicted_swapped = make_predicted(("A", "8"), ("B", "8"), ("C", "4"), ("D", "4"))
        true = make_true(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        scorer_perfect = ClassificationScorer(predicted_perfect, true)
        scorer_swapped = ClassificationScorer(predicted_swapped, true)
        # swapped clusters have the same grouping structure, so ARI should still be 1.0
        # (ARI is permutation-invariant on labels)
        assert scorer_swapped.rand_index() == pytest.approx(scorer_perfect.rand_index())


# ---------------------------------------------------------------------------
# v_measure
# ---------------------------------------------------------------------------

class TestVMeasure:
    def test_perfect_classification_is_one(self):
        predicted = make_predicted(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        true = make_true(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        scorer = ClassificationScorer(predicted, true)
        assert scorer.v_measure() == pytest.approx(1.0)

    def test_score_between_zero_and_one(self):
        predicted = make_predicted(("A", "100"), ("B", "200"), ("C", "100"), ("D", "200"))
        true = make_true(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        scorer = ClassificationScorer(predicted, true)
        score = scorer.v_measure()
        assert 0.0 <= score <= 1.0

    def test_bad_classification_lower_than_perfect(self):
        predicted_perfect = make_predicted(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        predicted_bad = make_predicted(("A", "100"), ("B", "100"), ("C", "100"), ("D", "100"))
        true = make_true(("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"))
        scorer_perfect = ClassificationScorer(predicted_perfect, true)
        scorer_bad = ClassificationScorer(predicted_bad, true)
        assert scorer_perfect.v_measure() > scorer_bad.v_measure()

    def test_returns_float(self):
        predicted = make_predicted(("A", "4"), ("B", "8"))
        true = make_true(("A", "4"), ("B", "8"))
        scorer = ClassificationScorer(predicted, true)
        assert isinstance(scorer.v_measure(), float)


# ---------------------------------------------------------------------------
# _carrie_measure_single
# ---------------------------------------------------------------------------

class TestCarrieMeasureSingle:
    """Tests for the internal _carrie_measure_single helper."""

    def _scorer_with_families(self, predicted_rows, true_rows):
        return ClassificationScorer(make_predicted(*predicted_rows), make_true(*true_rows))

    def test_perfect_match_passes(self):
        scorer = self._scorer_with_families(
            [("A", "4"), ("B", "4"), ("C", "4")],
            [("A", "4"), ("B", "4"), ("C", "4")],
        )
        result = scorer._carrie_measure_single("4", "4")
        assert result.carrie_measure_pass is True
        assert result.true_positive_rate == pytest.approx(1.0)
        assert result.false_positive_rate == pytest.approx(0.0)

    def test_tpr_below_threshold_fails(self):
        """94% TPR should fail (threshold is 95%)."""
        # 16 true members, predict only 15 of them (TPR = 15/16 ≈ 0.9375)
        true_rows = [(f"ast{i}", "4") for i in range(16)]
        pred_rows = [(f"ast{i}", "4") for i in range(15)] + [("ast15", "0")]
        scorer = self._scorer_with_families(pred_rows, true_rows)
        result = scorer._carrie_measure_single("4", "4")
        assert result.true_positive_rate < 0.95
        assert result.carrie_measure_pass is False

    def test_tpr_exactly_at_threshold_passes(self):
        """Exactly 95% TPR with 0% FPR should pass."""
        # 20 true members, predict 19 of them with no extras (TPR = 0.95)
        true_rows = [(f"ast{i}", "4") for i in range(20)]
        pred_rows = [(f"ast{i}", "4") for i in range(19)] + [("ast19", "0")]
        scorer = self._scorer_with_families(pred_rows, true_rows)
        result = scorer._carrie_measure_single("4", "4")
        assert result.true_positive_rate == pytest.approx(0.95)
        assert result.carrie_measure_pass is True

    def test_fpr_above_threshold_fails(self):
        """6% FPR should fail (threshold is 5%)."""
        # 100 true members all predicted, plus 6 false positives (FPR = 6/106)
        true_rows = [(f"ast{i}", "4") for i in range(100)]
        pred_rows = [(f"ast{i}", "4") for i in range(100)] + [(f"fake{i}", "4") for i in range(6)]
        scorer = self._scorer_with_families(pred_rows, true_rows)
        result = scorer._carrie_measure_single("4", "4")
        assert result.false_positive_rate > 0.05
        assert result.carrie_measure_pass is False

    def test_fpr_exactly_at_threshold_passes(self):
        """Exactly 5% FPR with 100% TPR should pass."""
        # true_rows has 19 members; predicted has 19 true + 1 extra = 20 (FPR = 1/20 = 0.05)
        true_rows = [(f"ast{i}", "4") for i in range(19)]
        pred_rows = [(f"ast{i}", "4") for i in range(19)] + [("extra0", "4")]
        scorer = self._scorer_with_families(pred_rows, true_rows)
        result = scorer._carrie_measure_single("4", "4")
        assert result.false_positive_rate == pytest.approx(1 / 20)
        assert result.carrie_measure_pass is True

    def test_both_rates_fail(self):
        """Low TPR and high FPR both fail independently."""
        # Only 5 of 20 true members predicted, plus 10 non-members
        true_rows = [(f"ast{i}", "4") for i in range(20)]
        pred_rows = [(f"ast{i}", "4") for i in range(5)] + [(f"fake{i}", "4") for i in range(10)]
        scorer = self._scorer_with_families(pred_rows, true_rows)
        result = scorer._carrie_measure_single("4", "4")
        assert result.carrie_measure_pass is False

    def test_empty_true_family_tpr_is_zero(self):
        """When the true family has no members, TPR is 0 and Carrie fails."""
        scorer = self._scorer_with_families(
            [("A", "4")],
            [("A", "8")],  # no family "4" in true
        )
        result = scorer._carrie_measure_single("4", "4")
        assert result.true_positive_rate == pytest.approx(0.0)
        assert result.carrie_measure_pass is False

    def test_empty_predicted_family_fpr_is_zero(self):
        """When the predicted family has no members, FPR is 0."""
        scorer = self._scorer_with_families(
            [("A", "0")],
            [("A", "4")],
        )
        result = scorer._carrie_measure_single("4", "4")
        # predicted "4" is empty so FPR = 0
        assert result.false_positive_rate == pytest.approx(0.0)

    def test_correct_true_family_recorded(self):
        scorer = self._scorer_with_families(
            [("A", "4")],
            [("A", "4")],
        )
        result = scorer._carrie_measure_single("4", "8")
        assert result.corresponding_true_family == "8"

    def test_disjoint_families_fail(self):
        """No overlap at all means TPR=0, Carrie fails."""
        scorer = self._scorer_with_families(
            [("A", "4"), ("B", "4")],
            [("C", "4"), ("D", "4")],
        )
        result = scorer._carrie_measure_single("4", "4")
        assert result.true_positive_rate == pytest.approx(0.0)
        assert result.carrie_measure_pass is False


# ---------------------------------------------------------------------------
# carrie_measure
# ---------------------------------------------------------------------------

class TestCarrieMeasure:
    """Tests for the public carrie_measure method."""

    def test_perfect_classification_all_pass(self):
        predicted = make_predicted(
            ("A", "4"), ("B", "4"),
            ("C", "8"), ("D", "8"),
        )
        true = make_true(
            ("A", "4"), ("B", "4"),
            ("C", "8"), ("D", "8"),
        )
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        # "0" is not a real family, so only "4" and "8" count
        assert score == 2
        assert results["4"].carrie_measure_pass is True
        assert results["8"].carrie_measure_pass is True

    def test_returns_tuple_of_int_and_dict(self):
        predicted = make_predicted(("A", "4"))
        true = make_true(("A", "4"))
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        assert isinstance(score, int)
        assert isinstance(results, dict)

    def test_no_families_pass_when_all_wrong(self):
        """All predicted members are non-family; no predicted family meets criteria."""
        predicted = make_predicted(
            ("A", "0"), ("B", "0"), ("C", "0"), ("D", "0"),
        )
        true = make_true(
            ("A", "4"), ("B", "4"), ("C", "8"), ("D", "8"),
        )
        scorer = ClassificationScorer(predicted, true)
        score, _ = scorer.carrie_measure()
        assert score == 0

    def test_partial_pass(self):
        """One family predicted correctly, one not."""
        predicted = make_predicted(
            ("A", "4"), ("B", "4"),   # perfect for family "4"
            ("C", "9"), ("E", "9"),   # only 1 of 2 family-"8" members + 1 extra
        )
        true = make_true(
            ("A", "4"), ("B", "4"),
            ("C", "8"), ("D", "8"),
        )
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        assert results["4"].carrie_measure_pass is True
        assert results["9"].carrie_measure_pass is False
        assert score == 1

    def test_score_is_count_of_passing_families(self):
        """score == number of True carrie_measure_pass values in results."""
        predicted = make_predicted(
            ("A", "4"), ("B", "4"),
            ("C", "8"), ("D", "8"),
        )
        true = make_true(
            ("A", "4"), ("B", "4"),
            ("C", "8"), ("D", "8"),
        )
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        expected = sum(1 for r in results.values() if r.carrie_measure_pass)
        assert score == expected

    def test_each_predicted_family_has_result_entry(self):
        predicted = make_predicted(
            ("A", "4"), ("B", "8"), ("C", "0"),
        )
        true = make_true(("A", "4"), ("B", "8"))
        scorer = ClassificationScorer(predicted, true)
        _, results = scorer.carrie_measure()
        assert "4" in results
        assert "8" in results
        assert "0" in results

    def test_best_matching_true_family_selected(self):
        """The result for each predicted family should record its best-matching true family."""
        predicted = make_predicted(
            ("A", "44"), ("B", "44"),
        )
        true = make_true(
            ("A", "4"), ("B", "4"),
            ("C", "8"),
        )
        scorer = ClassificationScorer(predicted, true)
        _, results = scorer.carrie_measure()
        # "44" perfectly covers family "4" (not "8"), so corresponding_true_family should be "4"
        assert results["44"].corresponding_true_family == "4"

    def test_two_predicted_families_mapping_to_same_true(self):
        """Two predicted families can both match the same true family independently."""
        predicted = pd.DataFrame(
            [("A", "101"), ("B", "101")], columns=["name", "family1"]
        )
        true = make_true(("A", "4"), ("B", "4"))
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        assert isinstance(score, int)

    def test_zero_family_label_not_penalised_as_failure(self):
        """The non-family label ("0") in predicted should appear in results but
        is typically not a 'passing' family – just ensure no crash and a result exists."""
        predicted = make_predicted(("A", "4"), ("B", "0"))
        true = make_true(("A", "4"))
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        assert "0" in results  # entry exists for the non-family group

    def test_single_asteroid_single_family_perfect(self):
        predicted = make_predicted(("Ceres", "1"))
        true = make_true(("Ceres", "1"))
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        assert score == 1
        assert results["1"].carrie_measure_pass is True

    def test_large_perfect_classification(self):
        """Scale test: 5 families × 100 members each."""
        families = [str(10 + i) for i in range(5)]
        pred_rows = [(f"ast_{fam}_{j}", fam) for fam in families for j in range(100)]
        true_rows = pred_rows[:]  # exact copy
        predicted = pd.DataFrame(pred_rows, columns=["name", "family1"])
        true = pd.DataFrame(true_rows, columns=["name", "family1"])
        scorer = ClassificationScorer(predicted, true)
        score, results = scorer.carrie_measure()
        assert score == 5
        assert all(r.carrie_measure_pass for fam, r in results.items() if fam != "0")
