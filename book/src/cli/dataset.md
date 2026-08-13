<!-- PCU: cli-dataset | contract: contracts/apr-lint-producers-v1.yaml -->

# apr dataset

Dataset inspection tools.

**Category**: Inspection

## Synopsis

```text
apr dataset audio-inspect <FILE> [--format json|text] [-o FILE] [--force]
```

## What `audio-inspect` measures

It decodes an uncompressed RIFF/WAVE file and reports the shape and amplitude
extrema it actually measured:

| Field | Meaning |
|-------|---------|
| `sample_rate` | Hz, from the `fmt ` chunk — never resampled |
| `channels` | channel count, from the `fmt ` chunk — never mixed down |
| `samples` | frames per channel (torchaudio's `num_frames`) |
| `min` / `max` | amplitude extrema over every decoded sample |
| `codec` | `pcm_u8`, `pcm_s16le`, `pcm_s24le`, `pcm_s32le` or `pcm_f32le` |

Integer PCM is normalised by the negative full-scale magnitude, the
`torchaudio.load(normalize=True)` convention. Float payloads are reported as
stored, so a float WAV that overshoots ±1 shows up as such.

A container or codec it cannot decode — FLAC, MP3, Ogg, ADPCM, a truncated
`data` chunk, an empty stream — is **refused** with a non-zero exit and a
message naming what was found. It never estimates.

## Example

<!-- example-cost: trivial -->
```bash
apr dataset audio-inspect --help
```

Producing the observation `apr audio-inspect-lint` reads:

<!-- example-cost: interactive -->
```bash
apr dataset audio-inspect clip.wav --format json -o audio.json
apr audio-inspect-lint --json-file audio.json --expected-sample-rate 16000
```

## Full help

Run `apr dataset audio-inspect --help` for the complete option list.

## See also

- Consumer: [`apr audio-inspect-lint`](./audio-inspect-lint.md)
- Source: [`crates/apr-cli/src/commands/audio_inspect.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/audio_inspect.rs)
- Contract: [`contracts/apr-lint-producers-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-lint-producers-v1.yaml)
