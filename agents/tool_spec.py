"""SCAD search, dispatch, and return tool definitions."""

def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ''
    for i, tool in enumerate(tools):
        assert tool['type'] == 'function'
        fn = tool['function']
        if i > 0:
            ret += '\n'
        ret += f'---- BEGIN FUNCTION #{i + 1}: {fn["name"]} ----\n'
        ret += f'Description: {fn["description"]}\n'

        if 'parameters' in fn:
            ret += 'Parameters:\n'
            properties = fn['parameters'].get('properties', {})
            required_params = set(fn['parameters'].get('required', []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                is_required = param_name in required_params
                param_status = 'required' if is_required else 'optional'
                param_type = param_info.get('type', 'string')

                desc = param_info.get('description', 'No description provided')

                if 'enum' in param_info:
                    enum_values = ', '.join(f'`{v}`' for v in param_info['enum'])
                    desc += f'\nAllowed values: [{enum_values}]'

                ret += (
                    f'  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n'
                )
        else:
            ret += 'No parameters are required for this function.\n'

        ret += f'---- END FUNCTION #{i + 1} ----\n'
    return ret


def search_tool():
    search = {
        'type': 'function',
        'function': {
            "name": "search",
            "description": "Performs a web search: supply a string 'query' and optional 'topk'. The tool retrieves the top 'topk' results (default 10) for the query, returning their docid, url, and document content (may be truncated based on token limits).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query string for the search."
                    },
                    "topk": {
                        "type": "integer",
                        "description": "Return the top k pages.",
                    }
                },
                "required": [
                    "query"
                ]
            }
        }
    }
    open_page = {
        'type': 'function',
        'function': {
            'name': 'open_page',
            'description': (
                "Open a page by docid or URL and return the complete content. "
                "Provide either 'docid' or 'url'; if both are provided, prefer 'docid'. "
                "The docid or URL must come from prior search tool results."
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'docid': {
                        'type': 'string',
                        'description': 'Document ID from search results to resolve and fetch.',
                    },
                    'url': {
                        'type': 'string',
                        'description': 'Absolute URL from search results to fetch.',
                    },
                },
                'required': [],
            },
        },
    }
    finish = {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': """Return the final result when you have a definitive answer or cannot progress further. Provide a concise answer plus a brief, evidence-grounded explanation.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {
                        'type': 'string',
                        'description': 'A succinct, final answer.',
                    },
                    'explanation': {
                        'type': 'string',
                        'description': 'A brief explanation for your final answer. For this section only, cite evidence documents inline by placing their docids in square brackets at the end of sentences (e.g., [20]). Do not include citations anywhere else.',
                    },
                    'confidence': {
                        'type': 'string',
                        'description': 'Confidence: your confidence score between 0% and 100% for your answer',
                    },
                },
                'required': ['answer', 'explanation'],
            },
        },
    }
    return [search, open_page, finish]


def branch_tool():
    branch = {
        'type': 'function',
        'function': {
            'name': 'branch',
            'description': """Create a sub-branch to execute a sub-task.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'description': {
                        'description': 'A concise 3-5 word identifier for the sub-task  (e.g., "Narrow down last 7 overs stats", "Code Exploration", "Test Creation").',
                        'type': 'string'
                    },
                    'prompt': {
                        'description': 'Clear, compact task prompt: state objectives and critical info to preserve in the response. Be brief and informative.',
                        'type': 'string'
                    },
                },
                'required': ['description', 'prompt'],
            },
        },
    }
    return_tool = {
        'type': 'function',
        'function': {
            'name': 'return',
            'description': """Finish the interaction when the sub task is complete OR if the assistant cannot proceed further with the task.""",
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'description': 'A comprehensive message describing sub task outcome.',
                    },
                },
                'required': ['message'],
            },
        },
    }
    return [branch, return_tool]


TOOL_PROMPT = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""
