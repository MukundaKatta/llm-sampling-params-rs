# llm-sampling-params

A small, dependency-light Rust crate providing a **fluent builder for LLM sampling parameters** — temperature, `top_p`, `top_k`, `max_tokens`, stop sequences, seed, and presence/frequency penalties. It serializes to a plain JSON object compatible with common provider conventions (Anthropic / OpenAI style).

## Features

- Ergonomic, chainable builder API (`SamplingParams::new().temperature(0.7).max_tokens(1024)`).
- Convenience presets: `SamplingParams::greedy()` (deterministic decoding) and `SamplingParams::creative()`.
- `to_json()` serialization that **omits unset fields**, so you only send the parameters you actually configured.
- `merge()` to layer one parameter set on top of another (overrides win where set) — useful for per-request overrides on top of defaults.
- Getters and an `is_default()` check for inspection.
- Only one dependency: [`serde_json`](https://crates.io/crates/serde_json).

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llm-sampling-params = "0.1"
```

## Usage

```rust
use llm_sampling_params::SamplingParams;

// Build a parameter set fluently.
let params = SamplingParams::new()
    .temperature(0.7)
    .top_p(0.95)
    .max_tokens(1024)
    .add_stop("<end>");

assert_eq!(params.get_temperature(), Some(0.7));

// Serialize to JSON — unset fields are omitted.
let body = params.to_json();
// { "temperature": 0.7, "top_p": 0.95, "max_tokens": 1024, "stop": ["<end>"] }
```

### Presets

```rust
use llm_sampling_params::SamplingParams;

let greedy = SamplingParams::greedy();     // temperature 0.0, top_p 1.0
let creative = SamplingParams::creative(); // temperature 1.0, top_p 0.95
```

### Merging defaults with overrides

```rust
use llm_sampling_params::SamplingParams;

let defaults = SamplingParams::new().temperature(0.5).max_tokens(100);
let overrides = SamplingParams::new().temperature(0.9).top_k(50);

let merged = defaults.merge(&overrides);
assert_eq!(merged.get_temperature(), Some(0.9)); // override wins
assert_eq!(merged.get_max_tokens(), Some(100));  // kept from defaults
assert_eq!(merged.get_top_k(), Some(50));        // added by override
```

## Supported parameters

| Parameter | Setter | Type |
| --- | --- | --- |
| Temperature | `temperature` | `f64` |
| Nucleus sampling | `top_p` | `f64` |
| Top-k sampling | `top_k` | `u32` |
| Max tokens | `max_tokens` | `u32` |
| Stop sequences | `stop` / `add_stop` | `Vec<String>` / `impl Into<String>` |
| Seed | `seed` | `u64` |
| Presence penalty | `presence_penalty` | `f64` |
| Frequency penalty | `frequency_penalty` | `f64` |

## Development

```sh
cargo build
cargo test
```

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** `serde_json`

## License

Licensed under the MIT License. See the `license` field in `Cargo.toml`.
