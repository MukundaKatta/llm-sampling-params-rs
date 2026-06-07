# llm-sampling-params

[![CI](https://github.com/MukundaKatta/llm-sampling-params-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-sampling-params-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, dependency-light, fluent builder for **LLM sampling parameters** in Rust.

It models the knobs that virtually every chat/completion API exposes —
`temperature`, `top_p`, `top_k`, `max_tokens`, `stop` sequences, `seed`, and the
`presence`/`frequency` penalties — and serializes them to a plain JSON object
whose field names follow the Anthropic / OpenAI conventions. Unset fields are
omitted, so request payloads stay minimal and the provider applies its own
defaults.

## Features

- Fluent, chainable builder (`SamplingParams::new().temperature(0.7).max_tokens(1024)`).
- Ready-made `greedy()` and `creative()` presets.
- `to_json()` / `from_json()` round-tripping that ignores unknown keys, so you
  can feed it a full request body and pull out just the sampling fields.
- `validate()` for catching out-of-range values (negative temperature, `top_p`
  above 1, zero `top_k`/`max_tokens`, non-finite floats, empty stop sequences)
  before they reach a provider.
- `merge()` for layering overrides on top of a base configuration.
- Zero `unsafe` code; the only dependency is `serde_json`.

## Installation

Add it to your `Cargo.toml`:

```toml
[dependencies]
llm-sampling-params = "0.1"
```

Or with cargo:

```sh
cargo add llm-sampling-params
```

## Usage

```rust
use llm_sampling_params::SamplingParams;

fn main() {
    // Build a configuration fluently.
    let params = SamplingParams::new()
        .temperature(0.7)
        .top_p(0.95)
        .max_tokens(1024)
        .add_stop("<end>")
        .add_stop("\n\nHuman:");

    // Fail fast on invalid values.
    params.validate().expect("parameters should be valid");

    // Serialize to a request-ready JSON object (unset fields are omitted).
    let body = params.to_json();
    println!("{}", serde_json::to_string_pretty(&body).unwrap());
    // {
    //   "temperature": 0.7,
    //   "top_p": 0.95,
    //   "max_tokens": 1024,
    //   "stop": ["<end>", "\n\nHuman:"]
    // }

    // Start from a preset and layer overrides on top.
    let base = SamplingParams::creative();
    let overrides = SamplingParams::new().max_tokens(256);
    let merged = base.merge(&overrides);
    assert_eq!(merged.get_temperature(), Some(1.0)); // from the preset
    assert_eq!(merged.get_max_tokens(), Some(256));  // from the override

    // Parse an existing payload back into a builder (unknown keys are ignored).
    let parsed = SamplingParams::from_json(&body).unwrap();
    assert_eq!(parsed.get_temperature(), Some(0.7));
}
```

## API overview

### Construction

| Method                    | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `SamplingParams::new()`   | Empty builder with no parameters set.                  |
| `SamplingParams::greedy()`| Preset: `temperature = 0.0`, `top_p = 1.0`.            |
| `SamplingParams::creative()` | Preset: `temperature = 1.0`, `top_p = 0.95`.       |
| `SamplingParams::from_json(&Value)` | Parse a JSON object; unknown keys ignored.   |

### Builder setters (chainable, take `self` by value)

`temperature(f64)`, `top_p(f64)`, `top_k(u32)`, `max_tokens(u32)`,
`stop(Vec<String>)`, `add_stop(impl Into<String>)`, `seed(u64)`,
`presence_penalty(f64)`, `frequency_penalty(f64)`.

### Getters

`get_temperature()`, `get_top_p()`, `get_top_k()`, `get_max_tokens()`,
`get_stop()`, `get_seed()`, `get_presence_penalty()`, `get_frequency_penalty()`.

### Other methods

| Method            | Description                                                       |
| ----------------- | ---------------------------------------------------------------- |
| `is_default()`    | `true` if no parameters have been set.                           |
| `validate()`      | `Result<(), ValidationError>` — checks every set field's range.  |
| `to_json()`       | Serialize to a `serde_json::Value` object (omits unset fields).  |
| `merge(&other)`   | Layer `other` on top; `other`'s set values win.                  |

### Errors

`from_json` and `validate` return `ValidationError`, which has two variants:
`OutOfRange { field, message }` and `TypeMismatch { field, message }`. It
implements `std::error::Error` and `Display`.

## License

Licensed under the [MIT License](LICENSE).
