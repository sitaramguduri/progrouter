import contextlib
import io
import unittest
from public_checks import prompt_checks


class PublicCheckTests(unittest.TestCase):
    def checks(self, doc):
        return prompt_checks('def f(x):\n    """' + doc + '"""\n')

    def run_checks(self, checks, function):
        with contextlib.redirect_stdout(io.StringIO()):
            for check in checks:
                exec(check, {'f': function})

    def test_outputless_comparison_is_an_assertion(self):
        checks = self.checks('Double x.\n>>> f(2) == 4\n')
        self.run_checks(checks, lambda x: 2*x)
        with self.assertRaises(AssertionError):
            self.run_checks(checks, lambda x: 0)

    def test_setup_and_explicit_outputs_are_preserved(self):
        checks = self.checks('Double x.\n>>> x = 2\n>>> f(x)\n4\n>>> f(3) == 6\n')
        self.run_checks(checks, lambda x: 2*x)
        with self.assertRaises(AssertionError):
            self.run_checks(checks, lambda x: x)

    def test_malformed_outputless_example_is_not_executed(self):
        checks = self.checks('Double x.\n>>> f(2) [1, 2]\n>>> f(2) == 4\n')
        self.run_checks(checks, lambda x: 2*x)
        self.assertEqual(self.checks('No oracle.\n>>> f(2) [1, 2]\n'), [])

    def test_assert_and_multiline_examples(self):
        checks = self.checks('Double x.\n>>> assert f(2) == 4\n>>> f(\n...     3)\n6\n')
        self.run_checks(checks, lambda x: 2*x)
