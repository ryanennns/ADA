import pytz

tool_get_current_time = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "enum": pytz.all_timezones,
                    "description": "The current timezone, e.g., 'America/New_York'. Defaults to UTC.",
                }
            },
            "required": []
        }
    }
}

tool_toggle_lights = {
    "type" : "function",
    "function": {
        "name": "toggle_lights",
        "description": "Toggle the lights on or off.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["on", "off"],
                    "description": "The desired state of the lights."
                }
            },
            "required": ["state"]
        }
    }
}