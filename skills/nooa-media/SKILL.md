---
name: nooa-media
description: Attach typed Image, Audio, Video, and PDF File inputs to NOOA Predict and CodeAct agents, use show() in captured cells, and troubleshoot explicit route-local media_inputs, text-only replay, MIME, URL, and attachment limits. Use for multimodal agent inputs, not media generation or inferred provider capability.
compatibility: nooa package
---

# Work with NOOA media

Read [the media guide](../../docs/media.md) for a complete alias-based image
example, constructors for every kind, and offline checks. Read
[model configuration](../../docs/model-configuration.md) and
[Connect](../../docs/model-connect.md) for route setup. Paths are relative to
this source checkout; a copied skill needs matching-version source/docs, not
assumed behavior from another release.

## Establish the boundary first

1. Identify the actual alias, registry file, model, endpoint, API style, client
   type, and transport. Preserve unrelated entries and credentials. Explicitly
   load a non-discovered file with `reload_registry(Path(...))`.
2. Ask what media may be sent and whether paid checks are authorized. An
   unattended task with no such consent must stay offline. Do not transmit user
   media, URLs, or signed query strings as a diagnostic convenience.
3. Separate three facts: a constructor made an object; a task/cell attached it;
   the outgoing route permits and actually accepts its format. None implies
   the next, and acceptance does not establish perceptual quality.

## Declare permission, never infer it

`media_inputs` belongs on an alias entry or client constructor. Allowed values
are exactly `image`, `audio`, `video`, `pdf`; omitted/null/empty means text-only.
Use a list (Python also accepts tuples), not a bare string. Duplicates and
unknown values are invalid; there is no `file` permission.

```yaml
models:
  vision:
    model_name: openai/your-vision-model
    client_type: completion
    api_base: https://gateway.example/v1
    api_key_env: MODEL_KEY
    max_tokens: 4096
    media_inputs: [image]
```

Replace placeholders with the user's documented route; this is unverified
manual permission. Do not infer capability from a name, catalogue, provider
family, or text baseline. `CompletionClient` and `ResponsesClient` accept
`media_inputs=[...]`; `get_llm_client("vision", media_inputs=["image", "pdf"])`
replaces the list for that client, without saving or merging it. Explicit `[]`
also silences the missing-declaration warning for an old alias.

Permissions do not follow replacement routes. Factory model/endpoint/provider/
client-type/client overrides clear inherited declarations unless explicitly
redeclared. Per-call route replacement defaults to text-only; credentials-only
changes retain permission. Build a new configured client for another route.
Never put `media_inputs` into a call's kwargs, `extra_body`, or reasoning levels.

## Pick the right attachment path

- Import `Image`, `Audio`, `Video`, `File` from `nooa` or `nooa.media` at module
  level. Declare direct typed parameters, such as `image: Image` or `clip: Audio`.
  Do not interpolate `{image}` into a docstring: arguments are already shown.
- **PredictStrategy:** use for extraction/classification with a structured
  return type. Direct Media-valued effective bound parameters are attached to
  the task once per parameter, including positional inputs without mirrored
  double counting. No recursive container inspection, content deduplication,
  or PIL/matplotlib auto-conversion; convert to typed bytes first.
- **Default CodeAct/CodeActV2:** `InspectInputsPrefill` auto-`show()`s direct
  Media parameters. Other inputs are printed and all inputs remain live Python
  variables. Nested list/dict values are not recursively attached. Disabling
  the plugin with `CodeActConfig(prefill=None)` or replacing it may require
  manual show; pre-ellipsis code still runs.
- **Generated CodeAct cells:** `show(image)` is a builtin, no import needed.
  Helpers in ordinary source import `from nooa.runtime.media_capture import
  show`; when invoked within the active cell they share capture. For a list,
  loop and show individual values. Strings/containers are not accepted by show.
- Outside the execution collector, `show()` only prints a not-captured notice.
  `print`, `pprint`, repr, returns, and assigning a Media to `self.context` are
  not attachment mechanisms. Do not rely on detached background tasks or a
  final result cell being followed by another model request.

## Use factories honestly

```python
from pathlib import Path
from nooa import Image, Audio, Video, File

photo = Image.from_file("photo.png")
clip = Audio.from_bytes(Path("clip.wav").read_bytes(), media_type="audio/wav")
movie = Video.from_file("clip.mp4")
report = File.from_bytes(Path("report.pdf").read_bytes(), media_type="application/pdf")
photo_ref = Image.from_url("https://example.org/photo.png", media_type="image/png")
```

All four classes support `from_file`, `from_bytes` (keyword MIME required), and
`from_url`. URL MIME defaults: Image `image/jpeg`, Audio `audio/wav`, Video
`video/mp4`, File `application/pdf`. File uses the `pdf` permission only for
exact PDF MIME hints. Non-PDF File attachments are removed even if every kind
is enabled: read/extract text locally for CSV/text/Word rather than claiming
arbitrary file support or relabeling bytes.

- Disk factories read all bytes and guess MIME by extension, falling back to
  octet-stream. WAV inference can yield `audio/x-wav`, which emits the literal
  audio format `x-wav`; it is not normalized to the checked `wav` format. Prefer
  explicit `audio/wav` bytes. Bytes factories do not validate content/codecs/MIME.
  URL factories store without fetching, validating, or sniffing. A `.pdf` name
  or file ID alone is not PDF evidence; conflicting/parameterized MIME hints
  are not exact `application/pdf`.
- **Audio.from_url does not fetch audio.** The current converter leaves a
  regular URL in `input_audio.data`, not a download-and-base64 path. Prefer
  valid local WAV bytes on a checked route; explicit fetching needs consent
  and bounds. Image/video/PDF URL handling also depends on endpoint support
  and accessibility. Do not promise arbitrary URLs or universal provider support.
- Optional PIL images and matplotlib Figures can be shown inside captured
  cells; lazy imports convert to PNG. Use `PILImage` as an import alias to avoid
  shadowing NOOA Image. `uv add pillow matplotlib` installs them if needed;
  ordinary Media and gating need neither library.
- Provider metadata is a hint, not a guarantee. The converter merges it for
  Image and Video, not Audio or File. The File block uses filename `attachment`.

## Bound size and explain replay

There is no constructor byte limit or universal endpoint limit. Check actual
byte/token/resolution/page/frame/duration/codec limits, resize or extract first,
and allow for base64's roughly 4:3 expansion and video frame costs. `size_bytes`
is approximate (None for URLs); text truncation does not compress attachments.

Normal actor capture defaults to 5 attachments **per cell**, including prefill;
configure `TruncationConfig(media_capture=MediaCaptureConfig(
max_attachments_per_execution=3))` via Agent `truncation=` (imports from
`nooa.config.truncation_config`). The positive count keeps first N atomically,
drops later shows with a limit notice, does not deduplicate, and converts before
checking count. It is not a memory cap. Predict task attachment collection does
not use it; sandbox `run_cell_source` currently has a separate cap of 100.

Successful capture on a text-only route is expected. The next request replaces
disabled recognized attachments with plain notices, for example:

- `[An image was removed here because this model is not set up to receive images.]`
- `[An audio clip was removed here because this model is not set up to receive audio.]`
- `[A video was removed here because this model is not set up to receive video.]`
- `[A PDF was removed here because this model is not set up to receive PDF files.]`
- `[A file was removed here. Only PDF files can be sent to a model.]`

Stored portable history and Media objects remain intact, so replay on a
permitted route can send the originals again. A notice does not transcribe or
describe content. Gating covers public attachment slots, not arbitrary nested
JSON, tool arguments, or a security scrub of reasoning/vendor state. Compatible
opaque native assistant media cannot be partially rewritten and may raise
`ReasoningReplayError`, requiring a portable assistant turn; not every native
replay can be promised placeholders.

## Choose offline versus consented checks

**Offline:** manual YAML or an unsaved plan makes no inference calls:

```sh
uv run nooa connect your-model --stage plan --as vision \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --media-inputs image,pdf --max-tokens 4096 > media-plan.json
```

Plan is offline; saving requires a separate authorized `--stage save` write.
Wizard `--media-inputs image,pdf --no-probe --no-catalogue` plus explicit route
avoids testing and catalogue access; `--no-probe` alone can still fetch metadata.
Use `--media-inputs ''` or `-` for text-only.

**Only with spending/sample-transmission approval:** use a bounded media stage,
not an unrelated full diagnostic sweep:

```sh
uv run nooa connect your-model --stage media --as vision \
  --endpoint https://gateway.example/v1 --api-style chat --api-key-env MODEL_KEY \
  --test-media-inputs image,pdf --max-tokens 4096 --budget-tokens 32768 > media-result.json
```

This can charge for the text baseline and generated samples. Stage mode does
not prompt or save implicitly; token budget is an estimate and failed calls
count. Wizard `--yes` bypasses approval prompts, approves checks and saving,
and can trigger billable calls: obtain user authorization before invoking it.
Stage checks run without a consent prompt regardless of that flag.
`--test-media-inputs` conflicts with manual
`--media-inputs`, `--no-probe`, and `--probe minimal` / `--probe none`; stage support is `plan`,
`media`, `all`. Planned checks are not executed checks. Audio/video checks are
Chat-only currently; other API styles skip them. Passing generated samples
establish route acceptance, not general understanding. Failed retests do not
automatically revoke prior manual permission. Read skipped/unverified results
and preserve declarations only for the same reconnect route.

## Verify and hand off

Inspect `src/nooa/media.py`, `runtime/media_capture.py`, `media_inputs.py`,
`unifiedllm/media_inputs.py`, `strategies/predict.py`, and `strategies/prefill.py`
under `src/nooa` for the installed source contract. Run the guide's offline
capture/gating/strategy tests, not live quickstarts, unless live calls were
approved. Check actual request shape with fake/mocked clients when possible.
Report alias/file, changes, what was tested, what remains unverified, and source
caveats; never report provider quality from mocked requests.

For migration, explicitly declare authorized media on older aliases. Prefer
typed parameters and captured show over hand-built dictionaries. Internal
`image_to_content_block` remains an alias of `media_to_content_block`; Media
has no `to_content_block()` method. This source has no `show_image()` or
`self.context.add_image()` API. Broad older docstrings such as File “PDF, etc.”
or guaranteed perception are qualified by current MIME/gating/adapter behavior.
