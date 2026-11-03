"""Transport constraints; workflow validators still enforce action semantics."""


def schema(kind, roles, allowed_actions=None):
    if kind.startswith('worker:'):
        return {'type': 'object', 'properties': {
            'message': {'type': 'string'},
            'code': {'type': 'string', 'minLength': 1, 'pattern': r'(^|\n)(async )?def [A-Za-z_][A-Za-z0-9_]*\s*\('} if kind == 'worker:coder' else {'type': ['string', 'null']},
            'run_tests': {'type': 'boolean'}},
            'required': ['message', 'code', 'run_tests'], 'additionalProperties': False}
    common = {'reason': {'type': 'string', 'minLength': 1},
              'plan': {'type': 'array', 'items': {'type': 'string'}},
              'state_summary': {'type': 'string'}}
    actions = {
        'dispatch': ({'role': {'type': 'string', 'enum': list(roles)},
                      'subtask': {'type': 'string', 'minLength': 1}}, ['role', 'subtask']),
        'finish': ({'final_answer': {'type': 'string', 'minLength': 1}}, ['final_answer']),
        'abort': ({'blocker': {'type': 'string', 'minLength': 1,
                            'description': 'An actual obstacle preventing completion, not a success statement'}}, ['blocker'])}
    branches = []
    for action in allowed_actions or actions:
        fields, required = actions[action]
        branches.append({'type': 'object',
            'properties': dict(common, action={'const': action}, **fields),
            'required': ['action', 'reason']+required, 'additionalProperties': False})
    return {'oneOf': branches}
