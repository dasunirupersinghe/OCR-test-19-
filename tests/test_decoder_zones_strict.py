"""Independent horizontal ownership and single-count joint scoring contracts."""
from dataclasses import replace
import math
import unittest

import numpy as np

from decoder import (CalibrationSample, DecoderError, DecoderOptions, DecoderPipeline,
                     HorizontalZoneBinder, ModifierCandidate, PrototypeCalibrationManager)
from features import CandidateIdentity, FeatureBatchResult
from segmentation import BoundingBox, DiacriticLink, SegmentationHypothesis
from tests.test_decoder import controlled_config, controlled_feature


def identity(cid, box, word_id=0):
    x0, y0, x1, y1 = box
    return CandidateIdentity(cid, word_id, tuple(box), tuple(box),
                             ((x0,y0),(x1,y0),(x1,y1),(x0,y1)), ())


def observed(cid, box, *, view, value=0., word_id=0):
    return replace(controlled_feature([value], view=view),
                   identity=identity(cid, box, word_id))


def mark(cid="m0", box=(12, 1, 16, 5), *, view="zone:upper", value=0., word_id=0):
    return ModifierCandidate(observed(cid, box, view=view, value=value, word_id=word_id))


def candidate(cid, box, value=0.):
    whole = observed(cid, box, view="whole_glyph", value=value)
    x0, _, x1, _ = box
    zones = {
        "core": observed(cid, (x0,20,x1,45), view="zone:core", value=value),
        "upper": observed(cid, (x0,1,x1,10), view="zone:upper"),
        "lower": observed(cid, (x0,50,x1,60), view="zone:lower"),
    }
    return replace(whole, zones=zones)


def batch_with(candidates, hypotheses=None, links=()):
    if hypotheses is None:
        hypotheses = (SegmentationHypothesis("partition", tuple(f.identity.candidate_id for f in candidates),
                                              "components", False),)
    return FeatureBatchResult(tuple(candidates), {f.identity.candidate_id:f for f in candidates},
                              {0:tuple(hypotheses)}, {0:tuple(links)}, {})


def models(*, upper="ි"):
    cfg = controlled_config()
    manager = PrototypeCalibrationManager(cfg)
    core = manager.fit((CalibrationSample("ක", controlled_feature([0.], view="zone:core")),
                        CalibrationSample("ග", controlled_feature([10.], view="zone:core"))))
    up = manager.fit((CalibrationSample(upper, controlled_feature([0.], view="zone:upper")),
                      CalibrationSample("ී", controlled_feature([10.], view="zone:upper"))))
    return cfg, {"zone:core":core, "zone:upper":up}


def pipeline(books, cfg):
    return DecoderPipeline(books, config=cfg,
                           options=DecoderOptions(recognition_mode="multi_zone"))


class HorizontalBinderContracts(unittest.TestCase):
    def setUp(self):
        self.cores = tuple(observed(cid, (left,20,right,45), view="zone:core")
                           for cid,left,right in (("c0",0,20),("c1",20,40)))

    def test_unique_overlap_uses_actual_feature_identity(self):
        bindings = HorizontalZoneBinder().bind(self.cores, (mark(),))
        self.assertEqual(bindings[0].parent_candidate_id, "c0")
        self.assertEqual(bindings[0].overlap_pixels, 4)
        self.assertEqual(bindings[0].overlap_fraction, 1.)

    def test_touching_boundary_has_zero_overlap(self):
        result = HorizontalZoneBinder().bind(self.cores, (mark(box=(20,1,24,5)),))
        self.assertEqual(result[0].parent_candidate_id, "c1")

    def test_unique_maximum_never_assigns_mark_twice(self):
        result = HorizontalZoneBinder().bind(self.cores, (mark(box=(18,1,24,5)),))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].parent_candidate_id, "c1")
        self.assertEqual(result[0].overlap_pixels, 4)
        self.assertAlmostEqual(result[0].overlap_fraction, 4/6)

    def test_equal_overlap_and_no_overlap_fail_explicitly(self):
        for box in ((18,1,22,5), (40,1,44,5)):
            with self.subTest(box=box), self.assertRaises(DecoderError):
                HorizontalZoneBinder().bind(self.cores, (mark(box=box),))

    def test_word_view_and_duplicate_identity_checks(self):
        with self.assertRaises(DecoderError):
            HorizontalZoneBinder().bind(self.cores, (mark(word_id=1),))
        with self.assertRaises(DecoderError):
            ModifierCandidate(observed("bad", (1,1,3,3), view="zone:core"))
        with self.assertRaises(DecoderError):
            HorizontalZoneBinder().bind(self.cores, (mark(), mark()))


class ExternalZonePipelineContracts(unittest.TestCase):
    def setUp(self):
        self.cfg, self.books = models()
        self.left = candidate("c0", (0,0,20,64), 0.)
        self.right = candidate("c1", (20,0,40,64), 10.)
        self.batch = batch_with((self.left,self.right))

    def test_external_mark_binds_left_without_native_crop_scoring(self):
        decoded = pipeline(self.books, self.cfg).run(self.batch, modifiers=(mark(),))
        self.assertEqual(decoded.text, "කිග")
        scores = decoded.words[0].scores
        self.assertEqual(set(scores), {"c0:zone:core", "c1:zone:core", "m0:zone:upper"})
        bindings = decoded.words[0].paths[0].metadata["modifier_bindings"]
        self.assertEqual(bindings[0]["parent_candidate_id"], "c0")

    def test_touch_and_strict_maximum_bind_right(self):
        for box in ((20,1,24,5),(18,1,24,5)):
            with self.subTest(box=box):
                decoded = pipeline(self.books,self.cfg).run(self.batch,modifiers=(mark(box=box),))
                self.assertEqual(decoded.text, "කගි")

    def test_ambiguous_split_rejected_and_intact_retained(self):
        whole = candidate("whole", (0,0,40,64))
        hypotheses = (SegmentationHypothesis("split", ("c0","c1"), "split", True),
                      SegmentationHypothesis("intact", ("whole",), "intact", True))
        batch = batch_with((whole,self.left,self.right), hypotheses)
        result = pipeline(self.books,self.cfg).run(batch,modifiers=(mark(box=(18,1,22,5)),))
        self.assertEqual(result.text,"කි")
        self.assertEqual({p.hypothesis_id for p in result.words[0].paths}, {"intact"})

    def test_unbound_or_wrong_word_mark_never_silently_disappears(self):
        result = pipeline(self.books,self.cfg).run(self.batch,modifiers=(mark(box=(40,1,44,5)),))
        self.assertEqual(result.words[0].status,"rejected")
        with self.assertRaises(DecoderError):
            pipeline(self.books,self.cfg).run(self.batch,modifiers=(mark(word_id=1),))

    def test_joint_log_probability_and_mean_max_confidence_count_mark_once(self):
        result = pipeline(self.books,self.cfg).run(self.batch,modifiers=(mark(),))
        word = result.words[0]
        path = word.paths[0]
        probs = {key:dict(zip(value.labels,value.probabilities)) for key,value in word.scores.items()}
        pleft = probs["c0:zone:core"]["ක"]
        pright = probs["c1:zone:core"]["ග"]
        pmark = probs["m0:zone:upper"]["ි"]
        self.assertAlmostEqual(path.log_probability,math.log(pleft)+math.log(pright)+math.log(pmark))
        self.assertAlmostEqual(path.confidence,(pleft*pmark+pright)/2)
        self.assertAlmostEqual(path.metadata["selected_label_confidence"],(pleft*pmark+pright)/2)

    def test_same_candidate_cache_varies_with_assigned_marks(self):
        farther = candidate("farther", (24,0,40,64), 10.)
        hypotheses = (SegmentationHypothesis("mark_right",("c0","c1"),"components",False),
                      SegmentationHypothesis("mark_left",("c0","farther"),"components",False))
        batch = batch_with((self.left,self.right,farther),hypotheses)
        result = pipeline(self.books,self.cfg).run(batch,modifiers=(mark(box=(18,1,24,5)),))
        by_hyp = {path.hypothesis_id:path.text for path in reversed(result.words[0].paths)}
        self.assertEqual(by_hyp["mark_right"],"කගි")
        self.assertEqual(by_hyp["mark_left"],"කිග")

    def test_explicit_empty_external_source_ignores_native_zone_ink(self):
        result = pipeline({"zone:core":self.books["zone:core"]},self.cfg).run(self.batch,modifiers=())
        self.assertEqual(result.text,"කග")

    def test_learned_null_modifier_keeps_probability_evidence(self):
        cfg, books = models(upper="")
        result = pipeline(books,cfg).run(self.batch,modifiers=(mark(),))
        self.assertEqual(result.text,"කග")
        self.assertIn("m0:zone:upper",result.words[0].scores)
        scores = result.words[0].scores
        pnull = dict(zip(scores["m0:zone:upper"].labels,scores["m0:zone:upper"].probabilities))[""]
        pc0 = dict(zip(scores["c0:zone:core"].labels,scores["c0:zone:core"].probabilities))["ක"]
        pc1 = dict(zip(scores["c1:zone:core"].labels,scores["c1:zone:core"].probabilities))["ග"]
        self.assertAlmostEqual(result.words[0].paths[0].log_probability,math.log(pc0)+math.log(pc1)+math.log(pnull))

    def test_native_geometry_ignores_stale_parent_ids(self):
        cfg, books = models()
        books["zone:lower"] = PrototypeCalibrationManager(cfg).fit((
            CalibrationSample("",controlled_feature([0.],view="zone:lower")),))
        whole = candidate("whole",(0,0,40,64))
        link = DiacriticLink("old","upper",BoundingBox(12,1,16,5),BoundingBox(12,1,16,5),("stale",),True)
        result = pipeline(books,cfg).run(batch_with((whole,),links=(link,)))
        self.assertEqual(result.text,"කි")
        self.assertEqual(result.words[0].paths[0].metadata["modifier_bindings"][0]["parent_candidate_id"],"whole")

    def test_external_window_ownership_uses_selected_covering_path(self):
        first = candidate("w0",(0,0,24,64))
        second = candidate("w1",(16,0,40,64),10.)
        hypotheses = (SegmentationHypothesis("windows",("w0","w1"),"sliding_windows",False),)
        result = pipeline(self.books,self.cfg).run(batch_with((first,second),hypotheses),
                                                   modifiers=(mark(box=(5,1,9,5)),))
        self.assertEqual(result.text,"කිග")
        self.assertEqual(result.words[0].paths[0].metadata["overlap_pixels"],8)
        self.assertEqual(result.words[0].paths[0].metadata["modifier_bindings"][0]["parent_candidate_id"],"w0")

    def test_both_external_zones_score_once_without_native_zone_models(self):
        books = dict(self.books)
        books["zone:lower"] = PrototypeCalibrationManager(self.cfg).fit((
            CalibrationSample("",controlled_feature([0.],view="zone:lower")),
            CalibrationSample("ු",controlled_feature([10.],view="zone:lower")),))
        result = pipeline(books,self.cfg).run(self.batch,modifiers=(
            mark(),mark("bottom",(12,50,16,55),view="zone:lower"),))
        self.assertEqual(result.text,"කිග")
        word = result.words[0]
        self.assertEqual(set(word.scores),{"c0:zone:core","c1:zone:core","m0:zone:upper","bottom:zone:lower"})
        labels = {"c0:zone:core":"ක","c1:zone:core":"ග","m0:zone:upper":"ි","bottom:zone:lower":""}
        expected = sum(math.log(dict(zip(result.labels,result.probabilities))[labels[key]])
                       for key,result in word.scores.items())
        self.assertAlmostEqual(word.paths[0].log_probability,expected)

    def test_native_nonoverlapping_zone_identity_rejected_without_links(self):
        cfg, books = models()
        books["zone:lower"] = PrototypeCalibrationManager(cfg).fit((
            CalibrationSample("",controlled_feature([0.],view="zone:lower")),))
        whole = candidate("whole",(0,0,40,64))
        detached_upper = observed("whole",(50,1,54,5),view="zone:upper")
        malformed = replace(whole,zones={**whole.zones,"upper":detached_upper})
        try:
            result = pipeline(books,cfg).run(batch_with((malformed,),links=()))
        except DecoderError:
            return
        self.assertEqual(result.words[0].status,"rejected")
        self.assertEqual(result.text,"")

    def test_multiple_native_mark_links_bind_once_but_zone_crop_scores_once(self):
        cfg, books = models()
        books["zone:lower"] = PrototypeCalibrationManager(cfg).fit((
            CalibrationSample("",controlled_feature([0.],view="zone:lower")),))
        whole = candidate("whole",(0,0,40,64))
        links = tuple(DiacriticLink(cid,"upper",BoundingBox(left,1,right,5),
                                    BoundingBox(left,1,right,5),("stale",),True)
                      for cid,left,right in (("first",12,16),("second",22,26)))
        result = pipeline(books,cfg).run(batch_with((whole,),links=links))
        self.assertEqual(result.text,"කි")
        bindings = result.words[0].paths[0].metadata["modifier_bindings"]
        self.assertEqual({entry["mark_id"] for entry in bindings},{"first","second"})
        self.assertEqual({entry["parent_candidate_id"] for entry in bindings},{"whole"})
        self.assertEqual(sum(key.endswith(":zone:upper") for key in result.words[0].scores),1)


if __name__ == "__main__":
    unittest.main()
