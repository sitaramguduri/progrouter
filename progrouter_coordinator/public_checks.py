"""Extract executable examples from public prompt docstrings without running code."""
import ast
import doctest


def prompt_checks(prompt):
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return []
    checks = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        try:
            examples = doctest.DocTestParser().get_examples(doc)
        except ValueError:
            continue
        # Preserve setup variables and order within a function's examples.
        if examples:
            checks.append("import doctest\n"
                "_doc = " + repr(doc) + "\n"
                "_test = doctest.DocTestParser().get_doctest(_doc, globals(), 'public', None, 0)\n"
                "_runner = doctest.DocTestRunner()\n"
                "_result = _runner.run(_test)\n"
                "assert _result.failed == 0, 'public doctest failed'\n")
    return checks
