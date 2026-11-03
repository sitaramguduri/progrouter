"""Transport constraints; workflow validators still enforce action semantics."""


def schema(kind, roles):
    if kind.startswith('worker:'):
        return {'type': 'object', 'properties': {
            'message': {'type': 'string'}, 'code': {'type': ['string', 'null']},
            'run_tests': {'type': 'boolean'}},
            'required': ['message', 'code', 'run_tests'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['dispatch', 'finish', 'abort']},
        'reason': {'type': 'string'}, 'role': {'type': 'string', 'enum': list(roles)},
        'subtask': {'type': 'string'}, 'plan': {'type': 'array', 'items': {'type': 'string'}},
        'state_summary': {'type': 'string'}, 'final_answer': {'type': 'string'}},
        'required': ['action', 'reason'], 'additionalProperties': False}
