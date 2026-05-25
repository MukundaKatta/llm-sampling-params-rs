/*!
llm-sampling-params: fluent builder for LLM sampling parameters.

Handles temperature, top_p, top_k, max_tokens, stop sequences, and more.
Serializes to a plain JSON object compatible with Anthropic/OpenAI conventions.

```rust
use llm_sampling_params::SamplingParams;

let p = SamplingParams::new()
    .temperature(0.7)
    .max_tokens(1024)
    .add_stop("<end>");
assert_eq!(p.get_temperature(), Some(0.7));
```
*/

use serde_json::{json, Value};

/// Fluent builder for LLM sampling parameters.
#[derive(Debug, Clone, Default)]
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
    pub fn new() -> Self { Self::default() }

    /// Preset for deterministic/greedy decoding.
    pub fn greedy() -> Self { Self::new().temperature(0.0).top_p(1.0) }

    /// Preset for creative outputs.
    pub fn creative() -> Self { Self::new().temperature(1.0).top_p(0.95) }

    // -- builder setters --
    pub fn temperature(mut self, v: f64) -> Self { self.temperature = Some(v); self }
    pub fn top_p(mut self, v: f64) -> Self { self.top_p = Some(v); self }
    pub fn top_k(mut self, v: u32) -> Self { self.top_k = Some(v); self }
    pub fn max_tokens(mut self, v: u32) -> Self { self.max_tokens = Some(v); self }
    pub fn stop(mut self, seqs: Vec<String>) -> Self { self.stop = seqs; self }
    pub fn add_stop(mut self, seq: impl Into<String>) -> Self { self.stop.push(seq.into()); self }
    pub fn seed(mut self, v: u64) -> Self { self.seed = Some(v); self }
    pub fn presence_penalty(mut self, v: f64) -> Self { self.presence_penalty = Some(v); self }
    pub fn frequency_penalty(mut self, v: f64) -> Self { self.frequency_penalty = Some(v); self }

    // -- getters --
    pub fn get_temperature(&self) -> Option<f64> { self.temperature }
    pub fn get_top_p(&self) -> Option<f64> { self.top_p }
    pub fn get_top_k(&self) -> Option<u32> { self.top_k }
    pub fn get_max_tokens(&self) -> Option<u32> { self.max_tokens }
    pub fn get_stop(&self) -> &[String] { &self.stop }
    pub fn get_seed(&self) -> Option<u64> { self.seed }
    pub fn get_presence_penalty(&self) -> Option<f64> { self.presence_penalty }
    pub fn get_frequency_penalty(&self) -> Option<f64> { self.frequency_penalty }

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

    /// Serialize to a JSON object (omits unset fields).
    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        if let Some(v) = self.temperature { m.insert("temperature".into(), json!(v)); }
        if let Some(v) = self.top_p { m.insert("top_p".into(), json!(v)); }
        if let Some(v) = self.top_k { m.insert("top_k".into(), json!(v)); }
        if let Some(v) = self.max_tokens { m.insert("max_tokens".into(), json!(v)); }
        if !self.stop.is_empty() { m.insert("stop".into(), json!(self.stop)); }
        if let Some(v) = self.seed { m.insert("seed".into(), json!(v)); }
        if let Some(v) = self.presence_penalty { m.insert("presence_penalty".into(), json!(v)); }
        if let Some(v) = self.frequency_penalty { m.insert("frequency_penalty".into(), json!(v)); }
        Value::Object(m)
    }

    /// Merge `other` on top; other's values win where set.
    pub fn merge(mut self, other: &SamplingParams) -> Self {
        if other.temperature.is_some() { self.temperature = other.temperature; }
        if other.top_p.is_some() { self.top_p = other.top_p; }
        if other.top_k.is_some() { self.top_k = other.top_k; }
        if other.max_tokens.is_some() { self.max_tokens = other.max_tokens; }
        if !other.stop.is_empty() { self.stop = other.stop.clone(); }
        if other.seed.is_some() { self.seed = other.seed; }
        if other.presence_penalty.is_some() { self.presence_penalty = other.presence_penalty; }
        if other.frequency_penalty.is_some() { self.frequency_penalty = other.frequency_penalty; }
        self
    }
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
        assert_eq!(SamplingParams::new().max_tokens(512).get_max_tokens(), Some(512));
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
            .temperature(0.8).top_p(0.9).max_tokens(256).add_stop("<end>");
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
    fn penalties_in_json() {
        let p = SamplingParams::new().presence_penalty(0.1).frequency_penalty(0.2);
        let v = p.to_json();
        assert!(v.get("presence_penalty").is_some());
        assert!(v.get("frequency_penalty").is_some());
    }

    #[test]
    fn is_not_default_after_set() {
        assert!(!SamplingParams::new().temperature(0.7).is_default());
    }
}
