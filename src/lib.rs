/*!
`llm-sampling-params`: a small, dependency-light, fluent builder for LLM sampling
parameters.

It handles the parameters that virtually every chat/completion API exposes —
`temperature`, `top_p`, `top_k`, `max_tokens`, `stop` sequences, `seed`, and the
`presence`/`frequency` penalties — and serializes them to a plain JSON object
whose field names follow the Anthropic / OpenAI conventions.

# Quick start

```rust
use llm_sampling_params::SamplingParams;

let p = SamplingParams::new()
    .temperature(0.7)
    .max_tokens(1024)
    .add_stop("<end>");

assert_eq!(p.get_temperature(), Some(0.7));
assert_eq!(p.to_json()["max_tokens"], 1024);
```

# Presets, merging and validation

```rust
use llm_sampling_params::SamplingParams;

// Start from a preset, then override individual fields.
let base = SamplingParams::creative();
let overrides = SamplingParams::new().max_tokens(256);
let merged = base.merge(&overrides);
assert_eq!(merged.get_temperature(), Some(1.0)); // from the preset
assert_eq!(merged.get_max_tokens(), Some(256)); // from the override

// Catch out-of-range values before sending them to a provider.
assert!(SamplingParams::new().temperature(0.7).validate().is_ok());
assert!(SamplingParams::new().top_p(1.5).validate().is_err());
```

# Round-tripping JSON

```rust
use llm_sampling_params::SamplingParams;

let original = SamplingParams::new().temperature(0.8).add_stop("STOP");
let json = original.to_json();
let parsed = SamplingParams::from_json(&json).unwrap();
assert_eq!(parsed.get_temperature(), Some(0.8));
assert_eq!(parsed.get_stop(), &["STOP".to_string()]);
```
*/

#![forbid(unsafe_code)]

use serde_json::{json, Value};
use std::fmt;

/// Error returned by [`SamplingParams::validate`] and
/// [`SamplingParams::from_json`] when a parameter is out of range or a value
/// has the wrong JSON type.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// A numeric parameter fell outside its allowed range.
    ///
    /// `field` is the parameter name, `message` explains the constraint that
    /// was violated.
    OutOfRange {
        /// Name of the offending parameter (e.g. `"temperature"`).
        field: &'static str,
        /// Human-readable explanation of the constraint.
        message: String,
    },
    /// A JSON value had a type that does not match the expected parameter type
    /// (only produced by [`SamplingParams::from_json`]).
    TypeMismatch {
        /// Name of the offending parameter.
        field: &'static str,
        /// Human-readable explanation of what was expected.
        message: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::OutOfRange { field, message } => {
                write!(f, "invalid value for `{field}`: {message}")
            }
            ValidationError::TypeMismatch { field, message } => {
                write!(f, "type mismatch for `{field}`: {message}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Fluent builder for LLM sampling parameters.
///
/// Every setter takes `self` by value and returns it, so calls can be chained.
/// Unset fields are simply omitted from [`to_json`](SamplingParams::to_json),
/// which keeps request payloads minimal and lets the provider apply its own
/// defaults.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SamplingParams {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    max_tokens: Option<u32>,
    stop: Vec<String>,
    seed: Option<u64>,
    presence_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
}

impl SamplingParams {
    /// Create an empty builder with no parameters set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Preset for deterministic / greedy decoding (`temperature = 0.0`,
    /// `top_p = 1.0`).
    pub fn greedy() -> Self {
        Self::new().temperature(0.0).top_p(1.0)
    }

    /// Preset for creative outputs (`temperature = 1.0`, `top_p = 0.95`).
    pub fn creative() -> Self {
        Self::new().temperature(1.0).top_p(0.95)
    }

    // -- builder setters --

    /// Set the sampling temperature. Higher values produce more random output.
    pub fn temperature(mut self, v: f64) -> Self {
        self.temperature = Some(v);
        self
    }

    /// Set nucleus sampling (`top_p`): keep the smallest set of tokens whose
    /// cumulative probability is at least `v`.
    pub fn top_p(mut self, v: f64) -> Self {
        self.top_p = Some(v);
        self
    }

    /// Set top-k sampling: sample only from the `v` most likely tokens.
    pub fn top_k(mut self, v: u32) -> Self {
        self.top_k = Some(v);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, v: u32) -> Self {
        self.max_tokens = Some(v);
        self
    }

    /// Replace the list of stop sequences.
    pub fn stop(mut self, seqs: Vec<String>) -> Self {
        self.stop = seqs;
        self
    }

    /// Append a single stop sequence to the existing list.
    pub fn add_stop(mut self, seq: impl Into<String>) -> Self {
        self.stop.push(seq.into());
        self
    }

    /// Set the random seed for reproducible sampling (where the provider
    /// supports it).
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = Some(v);
        self
    }

    /// Set the presence penalty.
    pub fn presence_penalty(mut self, v: f64) -> Self {
        self.presence_penalty = Some(v);
        self
    }

    /// Set the frequency penalty.
    pub fn frequency_penalty(mut self, v: f64) -> Self {
        self.frequency_penalty = Some(v);
        self
    }

    // -- getters --

    /// Get the configured temperature, if any.
    pub fn get_temperature(&self) -> Option<f64> {
        self.temperature
    }

    /// Get the configured `top_p`, if any.
    pub fn get_top_p(&self) -> Option<f64> {
        self.top_p
    }

    /// Get the configured `top_k`, if any.
    pub fn get_top_k(&self) -> Option<u32> {
        self.top_k
    }

    /// Get the configured `max_tokens`, if any.
    pub fn get_max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }

    /// Get the configured stop sequences (empty slice if none).
    pub fn get_stop(&self) -> &[String] {
        &self.stop
    }

    /// Get the configured seed, if any.
    pub fn get_seed(&self) -> Option<u64> {
        self.seed
    }

    /// Get the configured presence penalty, if any.
    pub fn get_presence_penalty(&self) -> Option<f64> {
        self.presence_penalty
    }

    /// Get the configured frequency penalty, if any.
    pub fn get_frequency_penalty(&self) -> Option<f64> {
        self.frequency_penalty
    }

    /// True if no parameters have been set.
    pub fn is_default(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.top_k.is_none()
            && self.max_tokens.is_none()
            && self.stop.is_empty()
            && self.seed.is_none()
            && self.presence_penalty.is_none()
            && self.frequency_penalty.is_none()
    }

    /// Validate that every set parameter falls within a commonly accepted range.
    ///
    /// The checks are intentionally lenient and provider-agnostic — they catch
    /// clearly invalid values (negative temperature, `top_p` above 1, `top_k`
    /// or `max_tokens` of zero, non-finite floats, empty stop sequences)
    /// without imposing a single vendor's exact limits.
    ///
    /// Returns the first error encountered, or `Ok(())` if everything is in
    /// range.
    ///
    /// ```
    /// use llm_sampling_params::SamplingParams;
    /// assert!(SamplingParams::new().temperature(0.5).validate().is_ok());
    /// assert!(SamplingParams::new().temperature(-0.1).validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), ValidationError> {
        if let Some(t) = self.temperature {
            if !t.is_finite() || t < 0.0 {
                return Err(ValidationError::OutOfRange {
                    field: "temperature",
                    message: format!("must be a finite value >= 0.0, got {t}"),
                });
            }
        }
        if let Some(p) = self.top_p {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(ValidationError::OutOfRange {
                    field: "top_p",
                    message: format!("must be within [0.0, 1.0], got {p}"),
                });
            }
        }
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err(ValidationError::OutOfRange {
                    field: "top_k",
                    message: "must be >= 1".to_string(),
                });
            }
        }
        if let Some(m) = self.max_tokens {
            if m == 0 {
                return Err(ValidationError::OutOfRange {
                    field: "max_tokens",
                    message: "must be >= 1".to_string(),
                });
            }
        }
        if let Some(pp) = self.presence_penalty {
            if !pp.is_finite() {
                return Err(ValidationError::OutOfRange {
                    field: "presence_penalty",
                    message: format!("must be finite, got {pp}"),
                });
            }
        }
        if let Some(fp) = self.frequency_penalty {
            if !fp.is_finite() {
                return Err(ValidationError::OutOfRange {
                    field: "frequency_penalty",
                    message: format!("must be finite, got {fp}"),
                });
            }
        }
        if self.stop.iter().any(|s| s.is_empty()) {
            return Err(ValidationError::OutOfRange {
                field: "stop",
                message: "stop sequences must not be empty".to_string(),
            });
        }
        Ok(())
    }

    /// Serialize to a JSON object (omits unset fields).
    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        if let Some(v) = self.temperature {
            m.insert("temperature".into(), json!(v));
        }
        if let Some(v) = self.top_p {
            m.insert("top_p".into(), json!(v));
        }
        if let Some(v) = self.top_k {
            m.insert("top_k".into(), json!(v));
        }
        if let Some(v) = self.max_tokens {
            m.insert("max_tokens".into(), json!(v));
        }
        if !self.stop.is_empty() {
            m.insert("stop".into(), json!(self.stop));
        }
        if let Some(v) = self.seed {
            m.insert("seed".into(), json!(v));
        }
        if let Some(v) = self.presence_penalty {
            m.insert("presence_penalty".into(), json!(v));
        }
        if let Some(v) = self.frequency_penalty {
            m.insert("frequency_penalty".into(), json!(v));
        }
        Value::Object(m)
    }

    /// Build a [`SamplingParams`] from a JSON object such as the one produced by
    /// [`to_json`](SamplingParams::to_json).
    ///
    /// Unknown keys are ignored, so this happily accepts a full request body and
    /// extracts just the sampling-related fields. Known keys with the wrong JSON
    /// type produce a [`ValidationError::TypeMismatch`].
    ///
    /// ```
    /// use llm_sampling_params::SamplingParams;
    /// use serde_json::json;
    ///
    /// let p = SamplingParams::from_json(&json!({
    ///     "temperature": 0.3,
    ///     "max_tokens": 200,
    ///     "stop": ["END"],
    ///     "model": "ignored"
    /// }))
    /// .unwrap();
    /// assert_eq!(p.get_temperature(), Some(0.3));
    /// assert_eq!(p.get_max_tokens(), Some(200));
    /// ```
    pub fn from_json(value: &Value) -> Result<Self, ValidationError> {
        let obj = value.as_object().ok_or(ValidationError::TypeMismatch {
            field: "<root>",
            message: "expected a JSON object".to_string(),
        })?;
        let mut p = SamplingParams::new();

        if let Some(v) = obj.get("temperature") {
            p.temperature = Some(as_f64(v, "temperature")?);
        }
        if let Some(v) = obj.get("top_p") {
            p.top_p = Some(as_f64(v, "top_p")?);
        }
        if let Some(v) = obj.get("top_k") {
            p.top_k = Some(as_u32(v, "top_k")?);
        }
        if let Some(v) = obj.get("max_tokens") {
            p.max_tokens = Some(as_u32(v, "max_tokens")?);
        }
        if let Some(v) = obj.get("seed") {
            p.seed = Some(as_u64(v, "seed")?);
        }
        if let Some(v) = obj.get("presence_penalty") {
            p.presence_penalty = Some(as_f64(v, "presence_penalty")?);
        }
        if let Some(v) = obj.get("frequency_penalty") {
            p.frequency_penalty = Some(as_f64(v, "frequency_penalty")?);
        }
        if let Some(v) = obj.get("stop") {
            p.stop = as_string_vec(v, "stop")?;
        }

        Ok(p)
    }

    /// Merge `other` on top; other's values win where set.
    pub fn merge(mut self, other: &SamplingParams) -> Self {
        if other.temperature.is_some() {
            self.temperature = other.temperature;
        }
        if other.top_p.is_some() {
            self.top_p = other.top_p;
        }
        if other.top_k.is_some() {
            self.top_k = other.top_k;
        }
        if other.max_tokens.is_some() {
            self.max_tokens = other.max_tokens;
        }
        if !other.stop.is_empty() {
            self.stop = other.stop.clone();
        }
        if other.seed.is_some() {
            self.seed = other.seed;
        }
        if other.presence_penalty.is_some() {
            self.presence_penalty = other.presence_penalty;
        }
        if other.frequency_penalty.is_some() {
            self.frequency_penalty = other.frequency_penalty;
        }
        self
    }
}

fn as_f64(v: &Value, field: &'static str) -> Result<f64, ValidationError> {
    v.as_f64().ok_or_else(|| ValidationError::TypeMismatch {
        field,
        message: format!("expected a number, got {v}"),
    })
}

fn as_u64(v: &Value, field: &'static str) -> Result<u64, ValidationError> {
    v.as_u64().ok_or_else(|| ValidationError::TypeMismatch {
        field,
        message: format!("expected a non-negative integer, got {v}"),
    })
}

fn as_u32(v: &Value, field: &'static str) -> Result<u32, ValidationError> {
    let n = as_u64(v, field)?;
    u32::try_from(n).map_err(|_| ValidationError::OutOfRange {
        field,
        message: format!("value {n} does not fit in a u32"),
    })
}

fn as_string_vec(v: &Value, field: &'static str) -> Result<Vec<String>, ValidationError> {
    let arr = v.as_array().ok_or_else(|| ValidationError::TypeMismatch {
        field,
        message: format!("expected an array of strings, got {v}"),
    })?;
    arr.iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| ValidationError::TypeMismatch {
                    field,
                    message: format!("expected a string element, got {item}"),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        assert!(SamplingParams::new().is_default());
    }

    #[test]
    fn temperature_roundtrip() {
        let p = SamplingParams::new().temperature(0.5);
        assert_eq!(p.get_temperature(), Some(0.5));
    }

    #[test]
    fn top_p_set() {
        assert_eq!(SamplingParams::new().top_p(0.9).get_top_p(), Some(0.9));
    }

    #[test]
    fn top_k_set() {
        assert_eq!(SamplingParams::new().top_k(40).get_top_k(), Some(40));
    }

    #[test]
    fn max_tokens_set() {
        assert_eq!(
            SamplingParams::new().max_tokens(512).get_max_tokens(),
            Some(512)
        );
    }

    #[test]
    fn stop_sequences() {
        let p = SamplingParams::new().stop(vec!["END".into(), "STOP".into()]);
        assert_eq!(p.get_stop().len(), 2);
    }

    #[test]
    fn add_stop_accumulates() {
        let p = SamplingParams::new().add_stop("A").add_stop("B");
        assert_eq!(p.get_stop().len(), 2);
    }

    #[test]
    fn seed_set() {
        assert_eq!(SamplingParams::new().seed(42).get_seed(), Some(42));
    }

    #[test]
    fn greedy_preset() {
        let p = SamplingParams::greedy();
        assert_eq!(p.get_temperature(), Some(0.0));
        assert_eq!(p.get_top_p(), Some(1.0));
    }

    #[test]
    fn creative_preset() {
        let p = SamplingParams::creative();
        assert_eq!(p.get_temperature(), Some(1.0));
        assert_eq!(p.get_top_p(), Some(0.95));
    }

    #[test]
    fn to_json_omits_unset() {
        let p = SamplingParams::new().temperature(0.7);
        let v = p.to_json();
        assert!(v.get("temperature").is_some());
        assert!(v.get("top_p").is_none());
    }

    #[test]
    fn to_json_full() {
        let p = SamplingParams::new()
            .temperature(0.8)
            .top_p(0.9)
            .max_tokens(256)
            .add_stop("<end>");
        let v = p.to_json();
        assert_eq!(v["temperature"], 0.8);
        assert_eq!(v["max_tokens"], 256);
        assert!(v["stop"].is_array());
    }

    #[test]
    fn merge_overrides() {
        let base = SamplingParams::new().temperature(0.5).max_tokens(100);
        let ov = SamplingParams::new().temperature(0.9).top_k(50);
        let merged = base.merge(&ov);
        assert_eq!(merged.get_temperature(), Some(0.9));
        assert_eq!(merged.get_max_tokens(), Some(100));
        assert_eq!(merged.get_top_k(), Some(50));
    }

    #[test]
    fn merge_keeps_base_stop_when_other_empty() {
        let base = SamplingParams::new().add_stop("A");
        let ov = SamplingParams::new().temperature(0.2);
        let merged = base.merge(&ov);
        assert_eq!(merged.get_stop(), &["A".to_string()]);
    }

    #[test]
    fn penalties_in_json() {
        let p = SamplingParams::new()
            .presence_penalty(0.1)
            .frequency_penalty(0.2);
        let v = p.to_json();
        assert!(v.get("presence_penalty").is_some());
        assert!(v.get("frequency_penalty").is_some());
    }

    #[test]
    fn is_not_default_after_set() {
        assert!(!SamplingParams::new().temperature(0.7).is_default());
    }

    // -- validation --

    #[test]
    fn validate_accepts_in_range() {
        let p = SamplingParams::new()
            .temperature(0.7)
            .top_p(0.95)
            .top_k(40)
            .max_tokens(256)
            .add_stop("END");
        assert!(p.validate().is_ok());
    }

    #[test]
    fn validate_rejects_negative_temperature() {
        let err = SamplingParams::new()
            .temperature(-0.5)
            .validate()
            .unwrap_err();
        assert!(matches!(
            err,
            ValidationError::OutOfRange {
                field: "temperature",
                ..
            }
        ));
    }

    #[test]
    fn validate_rejects_top_p_above_one() {
        assert!(SamplingParams::new().top_p(1.5).validate().is_err());
    }

    #[test]
    fn validate_rejects_top_p_below_zero() {
        assert!(SamplingParams::new().top_p(-0.01).validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_top_k() {
        assert!(SamplingParams::new().top_k(0).validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_max_tokens() {
        assert!(SamplingParams::new().max_tokens(0).validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_temperature() {
        assert!(SamplingParams::new()
            .temperature(f64::NAN)
            .validate()
            .is_err());
        assert!(SamplingParams::new()
            .temperature(f64::INFINITY)
            .validate()
            .is_err());
    }

    #[test]
    fn validate_rejects_empty_stop_sequence() {
        assert!(SamplingParams::new().add_stop("").validate().is_err());
    }

    #[test]
    fn validation_error_display_mentions_field() {
        let err = SamplingParams::new().top_p(2.0).validate().unwrap_err();
        assert!(err.to_string().contains("top_p"));
    }

    // -- from_json round-trips --

    #[test]
    fn from_json_round_trip() {
        let original = SamplingParams::new()
            .temperature(0.8)
            .top_p(0.9)
            .top_k(50)
            .max_tokens(123)
            .seed(7)
            .presence_penalty(0.1)
            .frequency_penalty(0.2)
            .add_stop("STOP")
            .add_stop("HALT");
        let parsed = SamplingParams::from_json(&original.to_json()).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn from_json_ignores_unknown_keys() {
        let v = json!({ "temperature": 0.3, "model": "claude", "extra": [1, 2, 3] });
        let p = SamplingParams::from_json(&v).unwrap();
        assert_eq!(p.get_temperature(), Some(0.3));
        assert!(p.get_max_tokens().is_none());
    }

    #[test]
    fn from_json_rejects_non_object() {
        assert!(SamplingParams::from_json(&json!([1, 2, 3])).is_err());
    }

    #[test]
    fn from_json_rejects_wrong_type() {
        let v = json!({ "temperature": "hot" });
        assert!(matches!(
            SamplingParams::from_json(&v),
            Err(ValidationError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn from_json_rejects_non_string_stop_element() {
        let v = json!({ "stop": ["ok", 5] });
        assert!(SamplingParams::from_json(&v).is_err());
    }

    #[test]
    fn from_json_empty_object_is_default() {
        let p = SamplingParams::from_json(&json!({})).unwrap();
        assert!(p.is_default());
    }
}
