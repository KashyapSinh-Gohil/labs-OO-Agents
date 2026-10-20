# Media inputs: images, audio, video, and PDFs

NOOA separates **creating media**, **attaching it to a task or execution
output**, and **allowing it on the outgoing model route**. Import the typed
values from `nooa` or `nooa.media`: `Image`, `Audio`, `Video`, and `File`.
These are provider-independent containers, not proof that an endpoint can
understand their contents.

## Declare permission on the route

Models are **text-only by default**. Set `media_inputs` on the registry entry
or client constructor, explicitly listing any of `image`, `audio`, `video`,
and `pdf`. Missing, null, and empty declarations all mean text-only. A list
(or a tuple in Python) is accepted; unknown values, duplicates, and a bare
string such as `"image"` are rejected. There is no `file` permission.

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

Replace the model and endpoint with your actual route and keep credentials in
the named environment variable. This configuration is an **unverified
permission**, not capability inference. NOOA does not enable media from a model
name, catalogue entry, provider family, or successful text call. The endpoint,
API style, transport, codecs, and request format must also support it.

For construction without a registry alias:

```python
from nooa.unifiedllm import CompletionClient, ResponsesClient

async def build_clients() -> None:
    chat = CompletionClient("openai/your-model", media_inputs=["image", "pdf"])
    responses = ResponsesClient("openai/your-model", media_inputs=["image"])
    try:
        # Creating clients sends no inference requests.
        print(chat.media_inputs, responses.media_inputs)
    finally:
        await chat.aclose()
        await responses.aclose()

# From an async application: await build_clients().
# aclose() releases sync and async resources; close() releases only sync ones.
```

`get_llm_client("vision", media_inputs=["image", "pdf"])` explicitly replaces
that client's configured list; it does not append to it or save YAML. Permission
is route-local: factory overrides of model, endpoint, provider, client type, or
client clear inherited permission unless explicitly redeclared for the new
client. Per-call model/endpoint/provider/client replacement uses text-only
permission. An API-key-only change does not clear it. Prefer a separately
configured client for each route. `media_inputs` is not a provider request
parameter: passing it per call, inside `extra_body`, or in reasoning-level
settings is invalid.

See [manual model configuration](model-configuration.md) for registry discovery
and [Connect](model-connect.md) for onboarding.

## Complete image example with a registry alias

Save the YAML above as `llm_config.yaml`, and this script as `inspect_image.py`
in the same directory. Use a real PNG at `photo.png` (in this checkout you can
copy `examples/assets/test_image.png`). Export `MODEL_KEY` securely. Run
`uv run python inspect_image.py` **only with permission to send the image and
make inference calls**; the two method calls below can incur charges. CodeAct
may use multiple turns; Predict may retry validation.

```python
import asyncio
from pathlib import Path

from pydantic import BaseModel

from nooa import Agent, Image, PredictStrategy, strategy
from nooa.unifiedllm import get_llm_client
from nooa.unifiedllm.registry import reload_registry

HERE = Path(__file__).resolve().parent
reload_registry(HERE / "llm_config.yaml")  # Explicit file, not assumed discovery.
llm = get_llm_client("vision")


class ImageFacts(BaseModel):
    description: str
    visible_text: list[str]


class Inspector(Agent, llm=llm):
    # Default CodeAct: input prefill attaches the Image via show(image).
    async def describe(self, image: Image) -> str:
        """Describe the image in one or two sentences. Do not invent details."""
        ...

    # Predict: typed media is attached directly to the task, with no tool loop.
    @strategy(PredictStrategy())
    async def extract(self, image: Image) -> ImageFacts:
        """Describe the image and extract legible text. Leave uncertain text out."""
        ...


async def main() -> None:
    try:
        image = Image.from_file(HERE / "photo.png")
        agent = Inspector()
        print(await agent.describe(image))
        print((await agent.extract(image)).model_dump_json(indent=2))
    finally:
        await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

Do not write `{image}` in the method docstrings: parameters are already
provided, and string interpolation is not a binary attachment mechanism.
Types used by generated code should be imported or defined at module level.

### Predict versus CodeAct inputs

- **PredictStrategy** attaches direct `Media` values in the effective bound
  parameters to its Task event. A positional input is not attached twice through
  its mirrored keyword binding. This is per-parameter collection, not content
  deduplication: distinct parameters holding the same object can attach it twice.
  Predict does not recursively walk lists/dicts or auto-convert PIL images and
  matplotlib figures. Use explicit typed parameters for extraction/classification.
- **Default CodeAct (also CodeActV2)** runs `InspectInputsPrefill`, which calls
  `show()` for direct Media-valued parameters and prints other values. The
  input objects are also live Python variables. Collection is shallow, not a
  recursive container walk, and the prefill has the normal per-cell attachment
  cap. `CodeActConfig(prefill=None)` disables this automatic inspection. With a
  custom/disabled prefill or media created later, explicitly call `show()` in
  generated code or a helper executing within the cell.

## `show()` in generated code and helpers

`show` is a builtin in generated CodeAct execution; no import is needed there.
It attaches one supported object to the current cell's output, for a subsequent
model request. It is not a general-purpose display function or URL downloader.
`print`, `pprint`, `repr`, and returning a Media object do not attach its bytes.
`show("photo.png")` and `show([image1, image2])` are unsupported; construct the
object or loop over the individual objects instead.

A hand-written helper imports `show` explicitly. For example, after the imports
and `llm` setup above, a CodeAct agent can define:

```python
from nooa.runtime.media_capture import show


class Gallery(Agent, llm=llm):
    def show_photo(self, path: str) -> None:
        """Attach a local photo during the current CodeAct cell."""
        show(Image.from_file(path))

    async def compare(self, paths: list[str]) -> str:
        """Call self.show_photo(path) for each path, then compare the photos."""
        ...
```

When generated code calls `self.show_photo(path)`, the helper shares the active
capture context. Calling the helper or `show(image)` from ordinary application
code **outside** an execution collector only prints
`[show() called outside execution context, not captured: ...]`. It does not
attach the object to a future call. Do not rely on detached background tasks
for capture. If `show()` is in the final cell that returns a result, there may
be no next model turn to inspect it.

### Optional PIL and matplotlib objects

Inside a captured cell, `show()` also accepts `PIL.Image.Image` and
`matplotlib.figure.Figure`, lazily converting them to PNG. Install the optional
libraries with `uv add pillow matplotlib` if your application needs them;
ordinary Media construction and permission filtering do not require them.
Use `from PIL import Image as PILImage` to avoid shadowing NOOA's `Image`.
For Predict, convert explicitly, for example save a PIL image into an
`io.BytesIO` as PNG and use `Image.from_bytes(buf.getvalue(),
media_type="image/png")`.

## Factories and focused examples

All four classes inherit `from_file(path, **vendor_metadata)` and
`from_bytes(data, *, media_type, **vendor_metadata)`. Each also has
`from_url(url, *, media_type=..., **vendor_metadata)`:

| Class | URL default MIME | Required permission | Capture block |
|---|---|---|---|
| `Image` | `image/jpeg` | `image` | `image_url` |
| `Audio` | `audio/wav` | `audio` | `input_audio` |
| `Video` | `video/mp4` | `video` | `video_url` |
| `File` | `application/pdf` | `pdf`, only for PDF MIME | `file` |

These snippets only construct values; the local files must exist and contain
valid media. Pass them as direct typed generation parameters or `show()` them
inside a captured CodeAct cell, on a suitably declared route.

```python
from pathlib import Path
from nooa import Image, Audio, Video, File

# Disk: reads all bytes now, guesses MIME from the filename extension.
photo = Image.from_file("photo.png")
clip = Audio.from_file("clip.wav")  # Inspect inferred MIME/format before sending.
movie = Video.from_file("clip.mp4")
report = File.from_file("report.pdf")

# Bytes: explicitly name the actual MIME; no codec/content validation.
photo_bytes = Image.from_bytes(Path("photo.png").read_bytes(), media_type="image/png")
clip_bytes = Audio.from_bytes(Path("clip.wav").read_bytes(), media_type="audio/wav")
movie_bytes = Video.from_bytes(Path("clip.mp4").read_bytes(), media_type="video/mp4")
report_bytes = File.from_bytes(Path("report.pdf").read_bytes(), media_type="application/pdf")

# References only: these calls make no HTTP requests.
photo_ref = Image.from_url("https://example.org/photo.png", media_type="image/png")
movie_ref = Video.from_url("https://example.org/clip.mp4", media_type="video/mp4")
report_ref = File.from_url("https://example.org/report.pdf", media_type="application/pdf")
clip_ref = Audio.from_url("https://example.org/clip.wav", media_type="audio/wav")
```

### URL, MIME, and transport caveats

- `from_url` stores the reference **without fetching or validating it**. For
  image/video/PDF references, endpoint fetching and accepted URL formats depend
  on the route. A public URL does not guarantee support; local paths and private
  URLs are not automatically accessible to the endpoint. Treat sending URLs,
  signed query strings, and media bytes as data disclosure requiring consent.
- **Audio URLs are not automatically downloaded.** The current converter emits
  `input_audio.data` from `_base64_data()`: a data URL gives base64, but a regular
  URL is left as the URL string in that field. This is not an implemented
  URL-to-audio fetch path and may be rejected. Prefer valid local WAV bytes/file
  on a route whose input-audio format you have checked; fetch remote audio
  explicitly in application code only with appropriate authorization and limits.
- `from_file` uses `mimetypes.guess_type`, falling back to
  `application/octet-stream`; it does not sniff the bytes. `from_bytes` requires
  the keyword `media_type` but does not verify that it matches the payload.
  URL defaults are labels, not detected MIME. Disk WAV inference can yield
  `audio/x-wav`; the converter takes the MIME suffix literally, emitting
  `input_audio.format="x-wav"` rather than normalizing it to `wav`. Prefer the
  explicit `audio/wav` bytes example for a route checked with that format.
  Constructors do not validate
  codecs, PDF validity, resolution, duration, URL reachability, or provider limits.
- The File attachment is PDF only when the collected MIME hints are exactly
  `application/pdf` (case-normalized). A `.pdf` filename/URL or file ID alone is
  not sufficient; missing, parameterized, or conflicting MIME hints do not pass
  as PDF. `File.from_url` supplies a default PDF hint, not PDF verification.
  **Non-PDF File attachments are replaced even with all four permissions.**
  For text/CSV/Word/etc., read or deterministically extract text locally and pass
  that text as a normal parameter. Do not relabel non-PDF bytes as PDF.
- Provider hints are not portable guarantees. The current converter merges
  `vendor_metadata` for Image (e.g. `detail="high"`) and Video, but not Audio or
  File. File blocks use the filename `attachment`. Adapters own final wire
  projection; enabling a kind does not ensure acceptance or perceptual accuracy.
  Video audio-track comprehension is also provider-dependent.

### Size and attachment limits

Factories read/encode the full payload. Base64 adds about 4:3 expansion; video
can have additional frame-token costs. There is no constructor byte-size cap or
universal provider limit here. Check your endpoint's byte, token, resolution,
page, frame, duration, codec, and URL rules; resize, sample, split, or extract
text before attaching. `size_bytes` is only an approximate inline payload size
(and `None` for regular URLs), not a validation result. Text truncation is not
binary media compression.

The normal actor runtime keeps at most **5 attachments per execution cell** by
default, including prefill. Configure
`TruncationConfig(media_capture=MediaCaptureConfig(max_attachments_per_execution=3))`
from `nooa.config.truncation_config` via an Agent's `truncation=` setting if
needed. The count must be positive. First N blocks are kept atomically; later
calls print `[show() limit reached (N), attachment not added]`. Repeated shows
are not deduplicated, and conversion happens before the count check, so this
is not a memory/byte/token cap. Predict's direct task attachments do not use
this per-cell cap. The sandbox `run_cell_source` currently uses a separate
hardcoded cap of 100; do not assume the actor's default applies to every backend.

## Text-only requests and replay

Capture can succeed even with `media_inputs: []`. When the next request is
built, disabled public attachments in fresh inputs, history, assistant content,
and tool outputs are replaced in place with plain text, preserving order:

- `[An image was removed here because this model is not set up to receive images.]`
- `[An audio clip was removed here because this model is not set up to receive audio.]`
- `[A video was removed here because this model is not set up to receive video.]`
- `[A PDF was removed here because this model is not set up to receive PDF files.]`
- `[A file was removed here. Only PDF files can be sent to a model.]`

Stored history and the Python Media objects are not changed. Replaying portable
history through a suitably permitted route can send its original attachments
again. The notice is neither a description nor OCR/transcription of the missing
media; a text-only model cannot recover its contents from it.

This is a policy on recognized attachment slots, **not a recursive security
scrubber** for tool arguments, reasoning, vendor extensions, or arbitrary nested
JSON. Compatible adapter-owned opaque assistant state cannot be partially
rewritten: disabled native media can raise `ReasoningReplayError` asking for a
portable assistant turn instead. Do not promise that every native replay turns
into a placeholder; incompatible native state follows the adapter's portable
fallback rules. See [assistant-turn replay](assistant-turn-replay.md).

## Offline configuration versus consented Connect checks

Offline planning constructs an unsaved, unverified proposal with no network:

```sh
uv run nooa connect your-model --stage plan --as vision \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --media-inputs image,pdf --max-tokens 4096 \
  > media-plan.json
```

Saving is a separate authorized write (`--stage save --input media-plan.json
--output llm_config.yaml`). Make that explicit file discoverable with
`NEMO_OO_LLM_CONFIG` or load it with `reload_registry(Path(...))`. In the wizard, manual `--media-inputs image,pdf
--no-probe --no-catalogue` with explicit model/endpoint/API style avoids probes
and catalogue access; `--no-probe` alone is not a blanket no-network guarantee
because discovery/catalogue may still fetch metadata. `--media-inputs ''` or
`-` explicitly selects text-only.

**Only after approval to spend and send generated samples**, a bounded stage
can run the configured text baseline plus chosen media checks:

```sh
uv run nooa connect your-model --stage media --as vision \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --test-media-inputs image,pdf \
  --max-tokens 4096 --budget-tokens 32768 > media-result.json
```

Stage mode does not ask for spending consent or save implicitly. The token
budget is an estimate, not a money cap; failed calls consume allowance and
checks can be skipped. In wizard mode, `--yes` bypasses approval prompts and
approves the configured checks and save, so it can trigger billable calls.
Obtain user authorization before invoking it; stage checks run without a
consent prompt regardless of `--yes`. `--test-media-inputs` conflicts with `--media-inputs`,
`--no-probe`, and `--probe minimal` / `--probe none`; in stage mode it is supported only by
`plan`, `media`, and `all`. Planning tests does not run them.

Generated-sample checks establish bounded request acceptance on the tested
route, not quality for arbitrary content. Image/PDF checks cover the three API
styles; audio/video checks currently run only for Chat, and are skipped on other
styles rather than proven unsupported. Passing formats can be enabled when the
result is applied/saved; failed retests do not automatically erase previous
manual permission. Omitted declarations are retained on reconnect only for the
same route; replacement routes default to text-only. Inspect checks, warnings,
provenance, and skipped/unverified results before saving. See
[Connect media setup](model-connect.md#choose-media-inputs-during-setup).

## Migration and source checks

For older aliases that previously relied on implicit media sending, explicitly
add only the verified/authorized kinds, or set `media_inputs: []` to remain
text-only and silence the once-per-alias missing-declaration warning. Do not
bulk-enable media based on model names.

Prefer typed method parameters and `show()` in captured cells over hand-built
message dictionaries. `image_to_content_block` remains an internal
backward-compatible alias of `media_to_content_block` in
`nooa.runtime.media_capture`; Media objects themselves have no
`to_content_block()` method. This checkout exposes no `show_image()` or
`self.context.add_image()` convenience API; do not copy those spellings from
older snippets. `self.context` manages prompt blocks, not the cell collector.

The source's older broad wording (File “PDF, etc.”, automatic provider
conversion, or `show` guaranteeing perception) must be read with the gating and
adapter limits above. For offline regression verification from this checkout:

```sh
uv run pytest -q -p no:cacheprovider tests/test_media.py tests/test_media_capture.py \
  tests/unifiedllm/test_media_inputs.py tests/runtime/test_show_media_filtering.py \
  tests/strategies/test_predict_media_dedup.py tests/unit/test_media_policy_imports.py
```

These tests use local/fake/mocked execution and requests, not paid provider
validation. Key contracts live in `src/nooa/media.py`,
`src/nooa/runtime/media_capture.py`, `src/nooa/media_inputs.py`,
`src/nooa/unifiedllm/media_inputs.py`, `src/nooa/strategies/predict.py`, and
`src/nooa/strategies/prefill.py`.
