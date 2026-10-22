"""손계산, 영벡터와 역색인 경계를 독립적으로 확인한다."""

import unittest
import numpy as np
from bow import NumpyTfidf, cosine, SearchIndex, BM25
from preprocess import sanitize, tokenize


class BowTests(unittest.TestCase):
    def test_tfidf_known_formula(self):
        v = NumpyTfidf(tf="raw", norm=None).fit(["good good bad", "bad movie"])
        self.assertAlmostEqual(v.idf_[v.vocabulary_["good"]], np.log(3 / 2) + 1)
        self.assertAlmostEqual(
            v.matrix_[0, v.vocabulary_["good"]], 2 * (np.log(3 / 2) + 1)
        )
        self.assertAlmostEqual(v.idf_[v.vocabulary_["bad"]], 1)

    def test_double_tf_and_zero_query(self):
        v = NumpyTfidf(tf="double", norm=None).fit(["good good bad", "movie"])
        self.assertAlmostEqual(
            v.matrix_[0, v.vocabulary_["bad"]] / v.idf_[v.vocabulary_["bad"]], 0.75
        )
        np.testing.assert_array_equal(
            v.transform(["unknownidentifier"]), np.zeros((1, 3))
        )
        np.testing.assert_array_equal(cosine(v.matrix_, np.zeros(3)), [0, 0])

    def test_cosine_geometry(self):
        np.testing.assert_allclose(
            cosine([[1, 0], [0, 1], [-1, 0]], [2, 0]), [1, 0, -1]
        )

    def test_inverted_exact_scores(self):
        index = SearchIndex(
            ["good movie good", "bad movie", "great story", "bad bad acting"]
        )
        for query in ("good movie", "bad acting", "unknownidentifier"):
            np.testing.assert_allclose(
                index.scores(query), index.scores(query, True), atol=1e-12
            )
            self.assertEqual(
                [i for i, _ in index.top(query)],
                [i for i, _ in index.top(query, inverted=True)],
            )

    def test_bm25_known_formula(self):
        v = BM25(["good good bad", "bad movie"])
        expected = np.log(2) * 5 / (2 + 1.5 * (0.25 + 0.75 * 3 / 2.5))
        self.assertAlmostEqual(v.scores("good")[0], expected)
        self.assertEqual(v.scores("good")[1], 0)

    def test_bm25_zero_document_at_b_one(self):
        model=BM25(['the','good'],b=1.)
        scores=model.scores('good')
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores[0],0)

    def test_search_methods_reject_nonpositive_result_count(self):
        for cls in (SearchIndex, BM25):
            index = cls(["good story", "bad acting"])
            for count in (0, -1):
                with self.assertRaises(ValueError):
                    index.top("good", count)

    def test_no_identifying_string_and_negation_retained(self):
        text = sanitize(
            "John Smith user@example.test 010-0000-1001 is not a good movie."
        )
        self.assertNotIn("john", text)
        self.assertNotIn("smith", text)
        self.assertNotIn("@", text)
        self.assertNotIn("010", text)
        self.assertIn("not", tokenize(text))
        self.assertIn("can't", tokenize(sanitize("I can't enjoy this movie.")))
        self.assertIn("good", tokenize(text))

    def test_test_word_does_not_change_training_idf(self):
        v = NumpyTfidf().fit(["good movie", "bad movie"])
        before = v.idf_.copy()
        v.transform(["excellent excellent excellent"])
        np.testing.assert_array_equal(before, v.idf_)
        self.assertNotIn("excellent", v.vocabulary_)
