"""Static checks only; candidate Python is never executed here."""
import ast


def expected_function(problem):
    prompt = problem.split('\nReturn the complete Python implementation', 1)[0]
    try:
        functions = [n.name for n in ast.parse(prompt).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return functions[-1] if functions else None
    except SyntaxError:
        return None


def validate_code(source, problem):
    if not isinstance(source, str) or not source.strip():
        return {'status': 'invalid', 'error': 'Provide the full Python function, not null or empty code.'}
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return {'status': 'invalid', 'error': f'SyntaxError at line {error.lineno}: {error.msg}',
                'line': (error.text or '').strip()[:250]}
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    target = expected_function(problem)
    selected = next((n for n in functions if target is None or n.name == target), None)
    if selected is None:
        return {'status': 'invalid', 'error': f'Provide a full top-level function {target or "implementation"}; placeholders such as "null" are not code.'}
    body = list(selected.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body or all(isinstance(n, ast.Pass) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is Ellipsis) for n in body):
        return {'status': 'invalid', 'error': 'Function body is a placeholder; provide its implementation.'}
    return {'status': 'valid', 'function': selected.name}
