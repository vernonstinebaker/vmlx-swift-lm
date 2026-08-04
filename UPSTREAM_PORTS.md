# Upstream Ports

This fork starts from `osaurus-ai/vmlx-swift-lm` commit
`4546a5d720e7013adffdbddd728c6106e4f9e637`, tagged as
`llmserverplus-vmlx-baseline-2026-08-04`.

No upstream commits have been imported yet. Each accepted port must be added
to this table with its prerequisite graph, conflict resolution, LLMServerPlus
test evidence, and upstream contribution URL when applicable.

| Official Commit | Subject | Parent Dependencies | Conflict Resolution | LLMServerPlus Evidence | Upstream Contribution |
| --- | --- | --- | --- | --- | --- |

## Abandoned Ports

### Gemma 4 unified native loading and unified-assistant MTP (2026-08-04)

The port was not landed. The required upstream graph is `67b146e`, `e145aca`,
`0767814`, `40c2ff0`, `09deb8c`, `68947cc`, and `78eaa5b` (with test follow-up
commits `eaefe75` and `83f3ef6`). It requires a manual rewrite of Gemma model
and factory integration, media position-ID handling, and upstream MTP runtime
paths that conflict semantically with this fork's BatchEngine/DFlash runtime.

The port was abandoned under the reconciliation contract rather than forcing a
merge. BatchEngine, CacheCoordinator, paged/disk cache diagnostics, and the
fork-native speculative runtime remain untouched. Re-evaluate only with a
bounded native unified-loader design that adapts to BatchEngine without
restoring the upstream MTP closure.
