import sys, unittest
sys.path.insert(0, "t4-local/agent")
from t4_agent.eps_parse import parse_pair, column_order

HDR_DESC = "Three Months Ended June 30, 2024 2023 "
HDR_ASC = "Three Months Ended September 30, Nine Months Ended September 30, 2022 2023 "


class T(unittest.TestCase):
    def pair(self, s):
        r = parse_pair(s)
        return None if r is None else (r[0], r[1])

    def test_basic_layouts(self):
        self.assertEqual(self.pair("Diluted $ 5.28 $ 2.68"), (5.28, 2.68))
        self.assertEqual(self.pair("Diluted $ ( 0.09 ) $ 0.56"), (-0.09, 0.56))
        self.assertEqual(self.pair("Diluted $ 8.62 $ 3.08 $ 20.21 $ 11.91"), (8.62, 3.08))
        self.assertEqual(self.pair("Diluted .97 .84 15.5 1.75 1.87 (6.4)"), (0.97, 0.84))

    def test_verified_six_token_layout_only(self):
        self.assertEqual(self.pair("Diluted earnings per share 6.12 4.75 29 10.56 8.85 19"), (6.12, 4.75))
        self.assertIsNone(self.pair("Diluted earnings per share 6.12 4.75 40 10.56 8.85 19"))      # percent column does not match

    def test_rejects_non_eps_rows(self):
        self.assertIsNone(self.pair("Diluted 1,611 1,410"))                                          # share counts
        self.assertIsNone(self.pair("Basic $ 3.34 $ 5.66"))                                          # basic EPS
        self.assertIsNone(self.pair("Reconciliation of adjusted earnings to GAAP Diluted $ 1.02 $ 0.33 $ 0.01 $ 0.01"))   # non-GAAP table
        self.assertIsNone(self.pair("Pro forma net loss per common unit: Basic $ (1.11) $ (0.98) Diluted $ (1.11) $ (0.98)"))  # pro forma
        self.assertIsNone(self.pair("Diluted 1.33 1.20 1.25 11 6 2.53 2.48 2"))                      # three-quarter layout
        self.assertIsNone(self.pair("Diluted $ 1.00 $ 2.00 and later Diluted $ 1.10 $ 2.00"))        # conflicting rows

    def test_column_order(self):
        self.assertEqual(column_order(HDR_DESC + "Diluted", len(HDR_DESC)), "desc")
        self.assertEqual(self.pair(HDR_DESC + "Diluted $ 4.39 $ 1.64"), (4.39, 1.64))
        self.assertEqual(column_order(HDR_ASC + "Diluted", len(HDR_ASC)), "asc")
        self.assertEqual(self.pair(HDR_ASC + "Diluted earnings per share $ 0.28 $ 0.94 $ ( 0.29 ) $ 1.89"), (0.94, 0.28))
        self.assertEqual(self.pair("Diluted $ 4.39 $ 1.64"), (4.39, 1.64))                            # no header: usual order

    def test_narrative(self):
        s = "Diluted earnings per share were $2.18, compared to $1.88 in the same quarter of the prior year."
        self.assertEqual(self.pair(s), (2.18, 1.88))
        self.assertIsNone(self.pair("Diluted earnings per share were $5.00, compared to $4.00 in the nine months ended September 30 of the prior year."))
        self.assertIsNone(self.pair("Diluted earnings per share were $5.00, compared to $4.00 for the fiscal year ended 2022."))

    def test_labels(self):
        self.assertEqual(self.pair("Earnings per share of common stock—assuming dilution $ 2.07 $ 1.64"), (2.07, 1.64))
        self.assertEqual(self.pair("Earnings per diluted common share $ 1.82 $ 1.24 $ 3.85 $ 2.95"), (1.82, 1.24))
        self.assertEqual(self.pair("Net income per share—diluted (4) $ 1.52 $ 1.33 $ 3.10 $ 3.52 (1)"), (1.52, 1.33))
        self.assertEqual(self.pair("Basic and diluted loss per share $ ( 1.23 ) $ ( 0.67 )"), (-1.23, -0.67))

    def test_year_to_date_may_be_smaller_after_a_loss_quarter(self):
        # real filings: an earlier loss quarter makes the year-to-date figure smaller than the quarter's (Ford, Marsh & McLennan)
        self.assertEqual(self.pair("Diluted 0.11 0.60 0.43 0.38"), (0.11, 0.60))

    def test_continuing_operations_rows_are_not_headline_eps(self):
        s = ("Diluted income from continuing operations per common share $ 0.47 $ 0.48 "
             "Diluted earnings per common share $ 0.50 $ 0.55")
        self.assertEqual(self.pair(s), (0.50, 0.55))
        self.assertIsNone(self.pair("Diluted earnings per share from continuing operations $ 0.47 $ 0.48"))   # only a continuing-operations row

    def test_corpus_replacement_character_in_labels(self):
        self.assertEqual(self.pair("Earnings per common share�diluted 4.42 2.32 8.05 4.22"), (4.42, 2.32))
        self.assertEqual(self.pair("Total diluted earnings (loss) attributable to shareholders $ 11.70 $ 1.97 $ 16.51 $ ( 1.38 )"), (11.70, 1.97))
        self.assertEqual(self.pair("Diluted income/(loss) 0.04 0.28 0.32 ( 0.22 )"), (0.04, 0.28))

    def test_tier_two_keeps_a_row_after_a_stray_adjusted_word(self):
        s = "Net income (adjusted for tax) 10 20 Net income per share—diluted (4) $ 1.52 $ 1.33 $ 3.10 $ 3.52 (1)"
        self.assertEqual(self.pair(s), (1.52, 1.33))

    def test_empty_and_garbage(self):
        for x in ("", None, 5, "no eps here", "Diluted"):
            self.assertIsNone(parse_pair(x))


if __name__ == "__main__":
    unittest.main(verbosity=1)
