# Media inputs: images, audio, video, and PDFs

Import `Image`, `Audio`, `Video`, and PDF `File` from `nooa.media` and pass them
as agent method arguments. For media created while CodeAct runs, use `show()`.

## Configure the model

NOOA assumes models are text-only until configured. Set `media_inputs` in the
model's registry entry or client constructor. Supported values are `image`,
`audio`, `video`, and `pdf`. Omitted, null, or empty lists mean text-only.
Use a list (or a tuple in Python); bare strings, duplicates, and unknown values
are rejected. If you enable a format the provider does not accept, requests may fail.

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

Replace placeholders with your provider's settings; keep the key in `MODEL_KEY`.
See [model configuration](model-configuration.md) and [Connect](model-connect.md).

## Complete image example

Save the YAML above as `llm_config.yaml` and this script as `inspect_image.py`
in the same directory. Put a real PNG at `photo.png`; this checkout includes
`examples/assets/test_image.png`. Export `MODEL_KEY` securely. Run
`uv run python inspect_image.py` only with approval to send the image and make
billable model calls. Predict may retry if the response fails validation.

```python
import asyncio
from pathlib import Path

from pydantic import BaseModel

from nooa import Agent, PredictStrategy, strategy
from nooa.media import Image
from nooa.unifiedllm import get_llm_client
from nooa.unifiedllm.registry import reload_registry

HERE = Path(__file__).resolve().parent
reload_registry(HERE / "llm_config.yaml")
vision = get_llm_client("vision")

class ImageFacts(BaseModel):
    description: str
    visible_text: list[str]

class Inspector(Agent, llm=vision):
    @strategy(PredictStrategy())
    async def extract(self, image: Image) -> ImageFacts:
        """Describe the image and extract legible text. Leave uncertain text out."""
        ...

async def main() -> None:
    try:
        image = Image.from_file(HERE / "photo.png")
        facts = await Inspector().extract(image)
        print(facts.model_dump_json(indent=2))
    finally:
        await vision.aclose()

if __name__ == "__main__":
    asyncio.run(main())
```

Predict suits extraction and classification: it attaches direct typed media
arguments and validates the return value. Import types at module level.
Leave `{image}` out of docstrings: arguments are already shown to the model.
For media in lists/dicts, use direct parameters or show each item in CodeAct.

## CodeAct and `show()`

Default CodeAct suits tasks that need Python or tools. Its input inspection
shows direct media parameters, prints other arguments, and keeps them as Python
variables. If you disable it with `CodeActConfig(prefill=None)` or replace it,
show media yourself.

In generated CodeAct, builtin `show(image)` attaches media from the cell for the
next model turn. Helpers import it explicitly. After the setup above, define:

```python
from nooa.runtime.media_capture import show

class Gallery(Agent, llm=vision):
    def show_photo(self, path: str) -> None:
        """Show a local photo while a CodeAct cell is running."""
        show(Image.from_file(path))

    async def compare(self, paths: list[str]) -> str:
        """Call self.show_photo for each path, then compare the photos."""
        ...
```

Call helpers from generated CodeAct cells. Outside a cell, `show()` prints a
not-captured notice. Show individual objects, not path strings or lists.
Use `show()` while generated code is running so the next model turn receives
the attachment. Printing the object displays its name and metadata.
`show()` also accepts PIL images and matplotlib figures, converting them to PNG;
install those optional libraries with `uv add pillow matplotlib` if needed.

## Create media values

All four types support `from_file(path)`, `from_bytes(data, *, media_type)`,
and `from_url(url, *, media_type=...)`:

```python
from pathlib import Path
from nooa.media import Image, Audio, Video, File

photo = Image.from_file("photo.png")
clip = Audio.from_bytes(Path("clip.wav").read_bytes(), media_type="audio/wav")
movie = Video.from_file("clip.mp4")
report = File.from_bytes(Path("report.pdf").read_bytes(), media_type="application/pdf")
photo_ref = Image.from_url("https://example.org/photo.png", media_type="image/png")
```

`from_file` reads the whole file and guesses MIME type from its extension, with
`application/octet-stream` as the fallback. `from_bytes` requires a MIME type.
These constructors do not check the content or codecs. WAV filenames can produce
`audio/x-wav`; prefer bytes with explicit `audio/wav` for providers expecting WAV.
URL default MIME types are `image/jpeg`, `audio/wav`, `video/mp4`, and
`application/pdf`, respectively.

`from_url` stores a URL without fetching it. Image, video, and PDF URLs need
provider support and must be accessible to the provider. **Audio URLs are not
automatically downloaded** and may be rejected; prefer local WAV bytes. Fetch
remote audio in application code only with authorization and size limits.

`File` supports PDFs only, with MIME type exactly `application/pdf`. Other file
attachments are removed, even with every format enabled. Read or extract text
locally from CSV, text, or Word files and pass it as a normal argument. Use the
actual MIME type; do not label other content as PDF. Video and audio format
support, including understanding a video's audio track, varies by provider.

## Size and attachment limits

Check your provider's byte, token, resolution, page, frame, duration, and codec
limits. Resize images, sample video, split PDFs, or extract text before attaching.
Constructors have no byte-size cap. Base64 adds about one third; video adds frame
tokens. `size_bytes` is approximate (`None` for URLs). Text truncation leaves
media bytes unchanged.

Default CodeAct capture keeps the first **5 attachments per cell**, including
input inspection. Later `show()` calls print a limit notice; repeated objects
count again. Set `TruncationConfig(media_capture=MediaCaptureConfig(
max_attachments_per_execution=3))` through the agent's `truncation=` setting to
change this positive count; both classes are in `nooa.config.truncation_config`.
Limit file sizes separately. Predict's direct arguments have no per-cell cap.

## Text-only requests and replay

For formats missing from `media_inputs`, NOOA replaces attachments with text
notices when sending the request, including attachments from earlier turns:

```text
[An image was removed here because this model is not set up to receive images.]
```

The notices contain no description or transcription. Stored history and Python
media objects stay unchanged. Reusing that history with a model configured for
the formats can send the original attachments again. Some saved provider-specific
responses instead raise `ReasoningReplayError` when media is disabled. In that
case, use the shared assistant-response format described in the
[replay guide](assistant-turn-replay.md).

## Connect checks and migration

Make an offline, unsaved configuration plan:

```sh
uv run nooa connect your-model --stage plan --as vision \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --media-inputs image,pdf --max-tokens 4096 > media-plan.json
```

Save separately with `--stage save --input media-plan.json --output llm_config.yaml`.
Load that file explicitly as above. In the wizard, use explicit model/endpoint/API
style with `--media-inputs image,pdf --no-probe --no-catalogue` to stay offline;
`--no-probe` alone can still fetch catalogue metadata.

With approval to spend and send generated samples, use `--stage media` with
`--test-media-inputs image,pdf --max-tokens 4096 --budget-tokens 32768`
instead of `--media-inputs`.
This runs a text baseline and media checks, can incur charges, and has no consent
prompt or automatic save. The token budget is an estimate; failed calls count.
Wizard `--yes` approves checks and saving. Audio/video checks currently run only
for Chat. Review passed, failed, and skipped checks before saving; see
[Connect media setup](model-connect.md#choose-media-inputs-during-setup).

For existing model entries, add the formats you use. Set `media_inputs: []`
for text-only models to silence the missing-list warning.

## Use a different model for one call

Pass a configured client with `llm=`. Its media settings apply to that call:

```python
vision = get_llm_client("vision")  # Configured with media_inputs: [image]
facts = await agent.extract(image, llm=vision)
```

Keep a separate alias for each model and endpoint. Changing the destination
inside a low-level call, such as `existing_client.acall(messages, model=...)`,
makes that request text-only. Use `llm=vision` on the agent method instead.

To change a client's formats without editing its saved entry, use
`get_llm_client("vision", media_inputs=["image", "pdf"])`.
Set `media_inputs` when configuring or constructing the client.
