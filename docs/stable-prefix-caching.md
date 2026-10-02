# Stable-prefix caching

NOOA places live context after stable instructions and conversation history.
One boundary identifies where that changing suffix begins. UnifiedLLM consumes
the boundary after projecting assistant turns into provider messages, so native
reasoning and multi-item Responses turns remain on the correct side.

This replaces positional rules such as "cache every system message and the last
tool result." Those rules can select changing context, and maintaining both
policies would make their interactions harder to test.

## Configuration

- `cache_breakpoint="auto"` is the default for both clients; registry entries need
  no cache setting. Completion clients mark recognized Anthropic routes with a
  native `cache_control` breakpoint; other Chat routes use provider-default caching.
  Responses clients enable explicit caching when a `CacheBoundary()` and eligible
  stable input are present. Without them, they send no extra cache fields.
- `cache_breakpoint="anthropic"` explicitly selects the Anthropic Chat mapping,
  including gateway aliases that cannot be recognized automatically.
- `cache_breakpoint="openai"` forces the Responses explicit-cache policy, including
  its leading-instructions fallback without a boundary. It is an override, not
  required for normal use.
- `cache_breakpoint=None` disables NOOA-generated cache markers, not the
  provider's implicit cache. Use this opt-out for a Responses endpoint that does
  not support explicit-cache fields. Responses clients do not accept the
  Anthropic Chat mapping.

Registry YAML accepts the same setting. Explicit mappings are tied to the client
model: use a new client when switching models. The automatic mapping is resolved
against the effective per-call model.

`CacheBoundary` belongs to the UnifiedLLM interface, alongside `LLMResponse`.
The cached renderer inserts it immediately before live context; the formatter
and middleware pass the same object through unchanged. Direct callers use it
in their message list too:

```python
from nooa.unifiedllm import CacheBoundary

messages = [*history, CacheBoundary(), {"role": "user", "content": live_state}]
response = await client.acall(messages)
```

Only UnifiedLLM interprets and removes the boundary before sending the request.
NeMo Relay projects it to public metadata JSON at its serialization boundary,
then restores the original object if that entry is unchanged. A raw dictionary
with `nooa_cache_boundary` is rejected with instructions to use `CacheBoundary()`;
there is only one accepted boundary type. Without a boundary, Completion and
forced explicit Responses policies mark only leading system/developer instructions,
not arbitrary history. Automatic Responses adds no cache fields.

## Provider mapping

Anthropic marks the latest eligible content block before the boundary, never a
thinking or redacted-thinking block. Images and documents are eligible, including
the `image_url` and `file` forms that LiteLLM converts for Anthropic. OpenAI
Responses reconstructs the latest 80 eligible message endpoints before the
boundary and enables explicit mode. Each endpoint is the message's last eligible
input text, image, file or function-result block; assistant output is skipped.
Stable media must be inside the breakpoint rather than left after a preceding
text block. This stateless fix retains recent checkpoints as history grows:
content stability alone is not enough if a warmed checkpoint is absent from the
next explicit request. OpenAI documents lookup at the latest 80 breakpoints and
writes at the latest four; four is not an annotation limit. If necessary,
stable Responses instructions become an input-text block to carry that marker.
If no eligible stable block exists, forced OpenAI explicit mode remains enabled with
no breakpoint and logs a warning: the request does not cache anything. This can
happen with wholly dynamic input or stable history containing only unmarkable
output blocks. It deliberately avoids implicit writes beyond the chosen boundary.
Automatic Responses instead leaves provider-default caching unchanged in this case.
Gemini receives no invented inline marker: this change uses its implicit cache,
not a separately managed explicit cached-content resource.

These mappings follow the [OpenAI prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching),
[Anthropic's content-block breakpoints](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
and [Gemini's implicit versus explicit caching](https://ai.google.dev/gemini-api/docs/caching).
OpenAI documents explicit mode and content-block breakpoints for GPT-5.6 and later;
gateway support must still be checked independently.

A boundary makes the stable prefix eligible for reuse; it does not guarantee a
hit. Provider thresholds, expiry, routing, model configuration and earlier edits
still matter. Changing effort or tools may invalidate an otherwise stable prefix.
Appending more than 80 new eligible endpoints in a single batch can exhaust the
lookup window and evict every previously warmed checkpoint; this bounded policy
does not promise a hit for that jump. The instructions-only to input-checkpoint
transition can also incur a one-time miss; that transition is not addressed by
this fix.

## Migration

`cache_control_injection_points` has been removed. Constructor and per-call use
raise a message naming `cache_breakpoint` and `CacheBoundary()` as replacements.
Use `cache_breakpoint=None` instead of an empty injection list. To cache completed
history, place a boundary after that history rather than selecting a message by
role or position.

`cache_breakpoint` belongs on the client constructor. Passing it through
`call`/`acall` or `extra_body` raises a configuration error before dispatch;
the framework setting must never become a provider request field.

## Code walkthrough: what changed and why

1. `unifiedllm/cache_policy.py` owns the single policy. It consumes the boundary
   and changes only marker targets' containers; unrelated messages and large
   strings are shared. The OpenAI reverse scan stops after 80 generated endpoints,
   skipping ineligible items, without content hashing, deep copies or persistent
   checkpoint state. Projection and boundary consumption still traverse history.
   `CacheBoundary` is a small, immutable UnifiedLLM input type. The cached renderer
   places it before live context using the same pass-through path as assistant
   responses. The formatter does not translate it or know what it means;
   ordinary messages carry no cache flag and are not edited.
2. Both clients apply the policy after provider projection. This keeps boundary
   placement correct when one stored assistant turn expands into several wire
   items, and keeps providers' fields out of renderers and middleware.
3. The former positional injection methods and their scenario tests are removed.
   Contract tests cover defaults, migration errors, ownership, dynamic context,
   native replay expansion and actual mocked HTTP payloads instead.
4. The opt-in live test closes SQLite, opens a new storage manager and client,
   changes trailing live state, and compares the stable HTTP prefix while
   checking native replay and reported cache tokens.

## Evidence and limits

### Offline request contracts

`tests/unifiedllm/test_history_wire_contract.py` runs on ordinary PR CI without
credentials. Synthetic provider replies pass through the real clients and SDKs;
an HTTP mock captures the outgoing requests and unexpected httpx requests fail.
It covers Responses encrypted reasoning, Anthropic signed thinking, Gemini tool
signatures on OpenAI-compatible routes, and Chat `reasoning_content`.

The matrix compares complete requests before and after rendering, SQLite
close/reopen, relay JSON reconciliation, and their combination, in both sync
and async calls. Only the trailing dynamic-context text is excluded. For
Anthropic that text can share a user message with tool results, which remain
inside the comparison. Tool schemas, instructions, argument strings, reasoning
fields and cache markers are checked, not just successful responses. Deliberate
edits and model/provider switches must instead discard incompatible native state
and retain readable reasoning. Negative controls corrupt captured reasoning,
arguments, instructions and tools and require the assertions to fail.

This follows [Pydantic AI's history-round-trip testing approach](https://github.com/pydantic/pydantic-ai/blob/8762545fa01fabfee6904e6166dd18aac759a005/tests/test_cache_prefix_stability.py), with assertions
on the requests generated by the current code rather than saved HTTP recordings.
It adds no runtime mechanism or test dependency. Run it with:

```bash
uv run pytest tests/unifiedllm/test_history_wire_contract.py
```

These tests prove request preservation, not provider cache hits. Growth tests
also check checkpoint retention and rollover beyond 80 endpoints. A synthetic
million-token-scale text history checks bounded markers and shared ownership.
The full CodeActV2 runtime HTTP test is parametrized with small and large fixed
`Context(prefix=True)` values, using one million `" alpha"` repetitions under the
explicit synthetic assumption of one token per repetition. It verifies complete
text preservation through rendering, `llm_call` middleware, Responses projection
and actual SDK serialization, SQLite close/reopen of historical assistant turns,
retained checkpoints over three turns, an excluded changing live suffix and no
input/native-state mutation. It uses a six-MB immutable string, not a tokenizer
benchmark or fragile timing/memory bound. Its synthetic context-block budget is
raised so the normal route limit does not evict the live suffix. No actual
1M-token live run is claimed.

The opt-in OpenAI-only growing-history live regression makes six requests at
79, 79, 80, 81, 82 and 82 eligible history endpoints, with a fixed model,
reasoning configuration and prompt-cache key, and `max_output_tokens=128`.
It checks exact marker rollover and retained old checkpoints, positive cache
reads on repeats/growth, and positive growth writes smaller than half the full
input count (a loose delta-write bound, not exact tokenizer accounting). It is
not run on ordinary CI and must be run on the deployment to validate real cache
lookup/writes. Adding the regression does not establish that its live assertions
have passed.

### Provider validation

Multimodal placement has regression tests through both serialized HTTP paths.
The opt-in resume test now ends the OpenAI and Anthropic stable prefixes with a
fixed image, and checks that the image itself carries the breakpoint. This adds
no calls to the existing three-call scenario. Offline preservation tests do not
establish cache support on a particular deployment.

```sh
uv run --extra nemo-relay pytest tests/unifiedllm/test_cache_policy.py tests/unifiedllm/test_explicit_cache_boundary.py
```

Live tests are opt-in and spend tokens. The resume suite is
`tests/integration/test_cache_resume_live.py`; it runs only when
`NOOA_RUN_CACHE_RESUME_LIVE=1` is set:

```sh
NOOA_RUN_CACHE_RESUME_LIVE=1 uv run pytest -m integration -s tests/integration/test_cache_resume_live.py
```

Configure its registry aliases for your deployment. Exact replay does not
control how much a provider caches; compare reported usage as well as outgoing
requests.

### Real CodeActV2 cache matrix

`tests/integration/test_agent_cache_live.py` exercises a module-level Agent
through the actual CodeActV2 runtime, rather than a standalone client prompt.
Its exact deployment matrix is:

| Case | Client | Model | API base |
| --- | --- | --- | --- |
| GPT 6.1 Sol | ResponsesClient | `openai/azure/openai/gpt-6.1-sol` | `https://inference-api.nvidia.com/v1` |
| Opus 5.5 | CompletionClient | `anthropic/azure/anthropic/claude-opus-5-5` | `https://inference-api.nvidia.com` |
| GLM 5.3 | CompletionClient | `openai/nvidia/zai-org/glm-5.3` | `https://inference-api.nvidia.com/v1` |
| Kimi K3 | CompletionClient | `openai/nvidia/moonshotai/kimi-k3` | `https://inference-api.nvidia.com/v1` |

A fixed `Context(prefix=True)` contains approximately 12K tokens of inert
padding, preceded by one unique per-run nonce that stays unchanged across turns
to prevent a previous trial from warming the large reference prefix. A private live-state attribute changes inside three executed cells
(`x=1`, increment, increment, printing each value); the fourth cell calls
`return_result(x)`. Expression Context is checked before every real model call.
The test demands result 3 and four valid, single `python_cell` calls, no
Python errors, and all four live phases in order. Each cell is AST-checked
against the required arithmetic step **before execution**; whitespace/comments
are allowed, but extra work or environment access fails without executing it.
Exactly four responses and outgoing HTTP POSTs are required; the eight-iteration
strategy budget is only a failure bound, not permission for hidden retries.
Historical assistant response objects are closed/reopened through a temporary
SQLite archive in `llm_call` middleware before subsequent dispatch; ordered
native parts, replay scope and usage must survive. This tests database-backed
assistant replay, not restoration of an interrupted execution session.

Usage is collected after the middleware continuation. At least two later calls
must report positive `cached_input_tokens`; the latest read must exceed half
of the initial input token count. The initial count must also exceed 8192.
Provider misses, omitted usage, invalid responses and provider errors **fail**:
there are no model/provider/cache-policy fallbacks or provider-dependent skips.
Output caps are 4096, HTTP timeout is 180 seconds, and transport/API retries
are zero. The CodeAct strategy uses an error budget of one (`max_retries=1`)
and permits at most eight iterations. Opus
uses adaptive thinking with low effort, GPT uses low reasoning effort, and GLM
and Kimi receive no speculative reasoning settings. Cache policy remains `auto`.

Ordinary CI runs only the deterministic HTTP matrix and cache-hit negative
controls. These use the same clients, SDK serialization, Agent and middleware,
and check native reasoning on outgoing mocked requests, caps, route identities
and supported thinking settings. They establish request contracts, **not** live
provider cache behavior:

```sh
uv run pytest tests/integration/test_agent_cache_live.py
```

To spend inference tokens, securely export `NVIDIA_INFERENCE_API_KEY`, then run:

```sh
NOOA_RUN_AGENT_CACHE_LIVE=1 uv run pytest -m integration -s tests/integration/test_agent_cache_live.py
```

Without opt-in the live cases skip (or are deselected by the default marker
filter). With opt-in, missing credentials fail. A live run normally makes 16
provider calls. A forwarding HTTP observer compares stable history and settings
in memory and reports only prefix-stability and call-count diagnostics; it does
not change requests or log raw payloads, credentials, signatures, encrypted
state or generated responses. Automatic trace export is disabled for
this module (the integration fixtures also reset tracing hooks/exporters).
SQLite archives and sidecars are deleted in cleanup, including failures;
while running they can contain native provider state. Do not publish them or
enable raw SDK/debug logging. Safe output is allowlisted route
configuration, normalized usage, and prefix-stability/call-count diagnostics only.

This belongs in pytest as the primary runtime regression. Connect probes test
endpoint capabilities; they are not a replacement for executing real CodeActV2
cells. Existing connect client/config construction may be reused in the future
if it becomes a stable, suitable public API; no production connect runner or
connect behavior change is required for this test.
