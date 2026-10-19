# -*- coding: utf-8 -*-
"""OpenAI (Responses) plugin-contract tests (plan §6.2 / A.3).

Single-turn tool branches (reasoning provider_state, call_id pairing),
pooled-client injection (D7), usage mapping (Q6) and neutral→native
Responses input conversion — fake response objects, no network.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from llm_provider_base import LlmProviderHandler

from llm_provider_openai import OpenAIResponsesProvider


def _agent(**config_overrides):
    configuration = {"model": "gpt-5"}
    configuration.update(config_overrides)
    return {
        "instructions": "You are terse.",
        "configuration": configuration,
        "llm": {"llm_name": "gpt"},
        "tool_call_role": "function_call_output",
    }


def _ctx(**overrides):
    defaults = {
        "http_client": httpx.Client(),
        "credentials": {
            "openai_api_key": "sk-openai",
            "api_key": "sk-openai",
        },
        "endpoint": "https://api.openai.com/v1",
        "model_identifier": "gpt-5",
        "options": {},
        "trace_id": "tid-1",
        "emit": None,
        "is_cancelled": None,
        "pool_name": "httpx_llm_openai",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _handler(ctx=None, agent=None):
    return OpenAIResponsesProvider(
        logging.getLogger("kit"), agent or _agent(), ctx=ctx or _ctx()
    )


def _usage(**overrides):
    defaults = {
        "input_tokens": 10,
        "output_tokens": 6,
        "total_tokens": 16,
        "input_tokens_details": SimpleNamespace(cached_tokens=2),
        "output_tokens_details": SimpleNamespace(reasoning_tokens=4),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _response(parts, usage=None, response_id="resp_1", status="completed"):
    return SimpleNamespace(
        id=response_id,
        status=status,
        usage=usage or _usage(),
        output=parts,
    )


# ---------------------------------------------------------------- metadata


def test_handler_is_plugin_base_subclass():
    assert issubclass(OpenAIResponsesProvider, LlmProviderHandler)


def test_plugin_metadata():
    assert OpenAIResponsesProvider.api_protocol == "openai-responses"
    assert OpenAIResponsesProvider.default_pool_name == "httpx_llm_openai"
    md = OpenAIResponsesProvider.plugin_metadata()
    assert isinstance(md["config_schema"], dict)
    # Q8/CREDS：凭据不入 schema
    assert "openai_api_key" not in md["config_schema"].get("properties", {})
    assert "openai_api_key" not in md["config_schema"].get("required", [])


def test_init_requires_pooled_client():
    ctx = _ctx()
    ctx.http_client = None
    with pytest.raises(Exception, match="ctx.http_client"):
        OpenAIResponsesProvider(logging.getLogger("k"), _agent(), ctx=ctx)


def test_init_excludes_credential_from_model_setting():
    handler = _handler(agent=_agent(openai_api_key="leaky"))
    assert "openai_api_key" not in handler.model_setting


# ------------------------------------------------- neutral → native (§5.5)


def test_to_native_drops_system_role():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert all(item.get("role") != "system" for item in native)
    assert native == [{"role": "user", "content": "hi"}]


def test_to_native_tool_shapes():
    handler = _handler()
    native = handler.to_native_messages(
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "find_order", "arguments": '{"q": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
    )
    assert native[1] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "find_order",
        "arguments": '{"q": 1}',
    }
    assert native[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "result",
    }


def test_to_native_provider_state_replay():
    handler = _handler()
    reasoning_item = {"type": "reasoning", "id": "rs_1", "summary": []}
    function_item = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "find_order",
        "arguments": "{}",
    }
    state = {
        "api_protocol": "openai-responses",
        "model": "gpt-5",
        "reasoning_item": reasoning_item,
        "items": [function_item],
    }
    native = handler.to_native_messages(
        [
            {"role": "assistant", "content": "", "provider_state": state},
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
    )
    assert native[0] == reasoning_item
    assert native[1] == function_item
    assert native[2]["type"] == "function_call_output"
    assert native[2]["call_id"] == "call_1"


def test_to_native_stale_state_dropped():
    handler = _handler()
    stale = {
        "api_protocol": "openai-responses",
        "model": "other-model",
        "items": [{"type": "function_call", "call_id": "x"}],
    }
    native = handler.to_native_messages(
        [{"role": "assistant", "content": "prior", "provider_state": stale}]
    )
    assert native == [{"role": "assistant", "content": "prior"}]


# ------------------------------------------- single-turn handle_response


def test_handle_response_tool_branch_single_turn():
    handler = _handler()
    reasoning_output = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        summary=[SimpleNamespace(text="musing")],
    )
    function_output = SimpleNamespace(
        type="function_call",
        call_id="call_9",
        name="find_order",
        arguments='{"q": "x"}',
        model_dump=lambda exclude_none=True: {
            "type": "function_call",
            "call_id": "call_9",
            "name": "find_order",
            "arguments": '{"q": "x"}',
        },
    )
    message_output = SimpleNamespace(
        type="message",
        status="completed",
        id="msg_1",
        role="assistant",
        content=[SimpleNamespace(text="checking.", annotations=[])],
    )
    response = _response(
        [reasoning_output, message_output, function_output],
        usage=_usage(input_tokens=12, output_tokens=5),
    )
    handler.handle_response(response, [])

    calls = handler.final_output["tool_calls"]
    assert calls == [
        {
            "id": "call_9",
            "type": "function",
            "function": {"name": "find_order", "arguments": '{"q": "x"}'},
        }
    ]
    state = handler.final_output["provider_state"]
    assert state["api_protocol"] == "openai-responses"
    assert state["items"][0]["type"] == "reasoning"
    assert state["items"][1] == {
        "type": "function_call",
        "call_id": "call_9",
        "name": "find_order",
        "arguments": '{"q": "x"}',
    }
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"]["total_tokens"] == 16
    assert handler.final_output["usage"]["cache_read_tokens"] == 2
    assert handler.final_output["usage"]["reasoning_tokens"] == 4
    assert handler.final_output["content"] == "checking."


def test_handle_response_plain_answer_usage_and_state():
    handler = _handler()
    reasoning_output = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        summary=[SimpleNamespace(text="m")],
    )
    expected_reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": reasoning_output.summary,
    }
    message_output = SimpleNamespace(
        type="message",
        status="completed",
        id="msg_2",
        role="assistant",
        content=[SimpleNamespace(text="answer", annotations=[])],
    )
    handler.handle_response(_response([reasoning_output, message_output]), [])
    assert handler.final_output["content"] == "answer"
    assert handler.final_output["finish_reason"] == "completed"
    assert handler.final_output["usage"]["total_tokens"] == 16
    state = handler.final_output["provider_state"]
    assert state["reasoning_item"] == expected_reasoning_item
    assert state["items"] == []


def test_handle_response_empty_retries():
    import unittest.mock as mock

    handler = _handler()
    responses = [
        _response(
            [
                SimpleNamespace(
                    type="message",
                    status="completed",
                    id="m0",
                    role="assistant",
                    content=[SimpleNamespace(text="", annotations=[])],
                )
            ]
        ),
        _response(
            [
                SimpleNamespace(
                    type="message",
                    status="completed",
                    id="m1",
                    role="assistant",
                    content=[SimpleNamespace(text="finally", annotations=[])],
                )
            ]
        ),
    ]

    def fake_invoke(**kwargs):
        return responses.pop(0)

    with mock.patch.object(handler, "invoke_model", side_effect=fake_invoke):
        handler.handle_response(responses[0], [])
    assert handler.final_output["content"] == "finally"


# --------------------------------------------- single-turn handle_stream


def _emit_capture(handler):
    emitted = []
    handler.ctx = SimpleNamespace(
        **{**handler.ctx.__dict__, "emit": lambda c: emitted.append(dict(c))}
    )
    return emitted


def test_handle_stream_tool_branch_single_turn():
    handler = _handler()
    emitted = _emit_capture(handler)
    handler.run = {"run_uuid": "run-1", "thread_uuid": "thread-1"}

    function_output = SimpleNamespace(
        type="function_call",
        call_id="call_5",
        name="lookup",
        arguments='{"q": 2}',
        model_dump=lambda exclude_none=True: {
            "type": "function_call",
            "call_id": "call_5",
            "name": "lookup",
            "arguments": '{"q": 2}',
        },
    )
    chunks = [
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="resp_9",
                usage=_usage(input_tokens=7, output_tokens=2),
                output=[function_output],
            ),
        ),
    ]
    finished = __import__("threading").Event()
    handler.handle_stream(iter(chunks), input_messages=[], stream_event=finished)

    assert finished.is_set()
    calls = handler.final_output["tool_calls"]
    assert calls[0]["id"] == "call_5"
    assert calls[0]["function"]["name"] == "lookup"
    assert handler.final_output["finish_reason"] == "tool_calls"
    assert handler.final_output["usage"]["total_tokens"] == 16
    assert handler.final_output["provider_state"]["items"][0]["call_id"] == "call_5"
    end_markers = [c for c in emitted if c.get("is_message_end")]
    assert end_markers, f"no message-end marker in {emitted!r}"


# ------------------------------------------ invoke_model params (2026-10-19 E2E fix)


def _responses_transport(captured):
    """Mock Responses API transport: capture the request body, return a valid response."""

    def _transport(request: "httpx.Request") -> "httpx.Response":
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-5-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "ok", "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 4,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 7,
                },
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "metadata": {},
                "temperature": 1.0,
                "top_p": 1.0,
                "tools": [],
                "reasoning": {"effort": None, "summary": None},
                "store": True,
                "text": {"format": {"type": "text"}},
            },
        )

    return httpx.MockTransport(_transport)


def test_invoke_model_maps_max_tokens_to_max_output_tokens():
    """Regression (E2E 2026-10-19): the adapter supplies the Chat-Completions-era
    ``max_tokens``; the Responses API rejects it with an unexpected-kwarg error."""
    captured = {}
    pooled = httpx.Client(transport=_responses_transport(captured))
    handler = _handler(
        agent=_agent(max_tokens=4096),
        ctx=_ctx(http_client=pooled),
    )
    handler.invoke_model(input=[{"role": "user", "content": "hi"}], stream=False)
    body = captured["body"]
    assert "max_tokens" not in body
    assert body.get("max_output_tokens") == 4096


def test_invoke_model_reasoning_drops_temperature():
    """Regression (E2E 2026-10-19): reasoning models reject non-default
    temperature ("Unsupported parameter: 'temperature' is not supported")."""
    captured = {}
    pooled = httpx.Client(transport=_responses_transport(captured))
    handler = _handler(
        agent=_agent(
            temperature=0.7,
            reasoning={"effort": "low", "summary": "required"},
        ),
        ctx=_ctx(http_client=pooled),
    )
    handler.invoke_model(input=[{"role": "user", "content": "hi"}], stream=False)
    body = captured["body"]
    assert body.get("reasoning") == {"effort": "low", "summary": "required"}
    assert "temperature" not in body


def test_invoke_model_passes_temperature_without_reasoning():
    captured = {}
    pooled = httpx.Client(transport=_responses_transport(captured))
    handler = _handler(
        agent=_agent(temperature=0.7),
        ctx=_ctx(http_client=pooled),
    )
    handler.invoke_model(input=[{"role": "user", "content": "hi"}], stream=False)
    assert captured["body"].get("temperature") == 0.7


def test_invoke_model_converts_neutral_tools_to_flat():
    """Regression (E2E 2026-10-19): neutral Chat-Completions-shaped tools must
    become the Responses flat function shape ("tools[0].name"), not 400."""
    captured = {}
    pooled = httpx.Client(transport=_responses_transport(captured))
    handler = _handler(
        agent=_agent(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        ),
        ctx=_ctx(http_client=pooled),
    )
    handler.invoke_model(input=[{"role": "user", "content": "hi"}], stream=False)
    tools = captured["body"]["tools"]
    assert tools == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_invoke_model_keeps_provider_shaped_tools():
    captured = {}
    pooled = httpx.Client(transport=_responses_transport(captured))
    handler = _handler(
        agent=_agent(tools=[{"type": "web_search"}]),
        ctx=_ctx(http_client=pooled),
    )
    handler.invoke_model(input=[{"role": "user", "content": "hi"}], stream=False)
    assert captured["body"]["tools"] == [{"type": "web_search"}]
