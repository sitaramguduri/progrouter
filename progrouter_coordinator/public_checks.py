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
        # Output-less comparisons in benchmark prose mean assertions, not
        # interactive expressions expected to print nothing. Keep only examples
        # with an explicit oracle, assertions, or setup statements. In particular,
        # malformed output-less prose such as f(x) [1, 2] is not a test oracle.
        normalized = []
        has_check = False
        for example in examples:
            try:
                statements = ast.parse(example.source).body
            except SyntaxError:
                continue
            if not example.want.strip():
                if len(statements) == 1 and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Compare):
                    example.source = ast.unparse(ast.Assert(test=statements[0].value)) + '\n'
                    has_check = True
                elif statements and all(isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Import, ast.ImportFrom, ast.Assert)) for n in statements):
                    has_check |= any(isinstance(n, ast.Assert) for n in statements)
                else:
                    continue
            else:
                has_check = True
            normalized.append(example)
        # Preserve setup variables and order within a function's examples.
        if has_check:
            doc = ''.join('>>> ' + e.source.rstrip().replace('\n', '\n... ') + '\n' + e.want for e in normalized)
            checks.append("import doctest\n"
                "_doc = " + repr(doc) + "\n"
                "_test = doctest.DocTestParser().get_doctest(_doc, globals(), 'public', None, 0)\n"
                "_runner = doctest.DocTestRunner()\n"
                "_result = _runner.run(_test)\n"
                "assert _result.failed == 0, 'public doctest failed'\n")
    return checks
