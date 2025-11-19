import unittest

from src.retrieval.citation.extract import extract_citations


class TestCitationExtract(unittest.TestCase):
    def test_id_based_nd_cp(self):
        text = "Theo khoản 2 Điều 14 Nghị định 91/2017/NĐ-CP quy định ..."
        hits = extract_citations(text)
        self.assertTrue(len(hits) >= 1)
        h = hits[0]
        self.assertEqual(h.method, "ID")
        self.assertEqual(h.act_code_norm, "91/2017/ND-CP")
        self.assertEqual(h.article, 14)
        self.assertEqual(h.clause, 2)

    def test_id_based_tt_bca(self):
        text = "Căn cứ Điều 8 Thông tư 104/2016/TT-BCA về ..."
        hits = extract_citations(text)
        code_norms = {h.act_code_norm for h in hits}
        self.assertIn("104/2016/TT-BCA", code_norms)
        # Ensure article is detected
        self.assertTrue(any(h.article == 8 for h in hits))

    def test_out_of_order(self):
        text = "Nghị định 91/2017/NĐ-CP quy định tại Điều 14 về ..."
        hits = extract_citations(text)
        self.assertTrue(any(h.act_code_norm == "91/2017/ND-CP" and h.article == 14 for h in hits))

    def test_name_based_ambiguous(self):
        text = "Theo Bộ luật Dân sự 2015, khoản 1 điều 3 quy định ..."
        hits = extract_citations(text)
        # Should produce a NAME hit with ambiguity True (no registry resolution yet)
        self.assertTrue(any(h.method == "NAME" and h.ambiguity for h in hits))


if __name__ == "__main__":
    unittest.main()
