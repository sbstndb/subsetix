import unittest

from subsetix_amr2.demo_two_level_advection import create_argparser


class DemoParserTest(unittest.TestCase):
    def test_defaults(self) -> None:
        parser = create_argparser()
        ns = parser.parse_args([])
        self.assertEqual(ns.coarse, 96)
        self.assertEqual(ns.steps, 300)
