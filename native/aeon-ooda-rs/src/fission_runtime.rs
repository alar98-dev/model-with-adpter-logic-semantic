//! Pure wave-fission primitives.  No Tokio, Redis, model or database work is
//! allowed here: callers can run these calculations off the streaming path.

#![allow(dead_code)]

use std::collections::HashSet;

pub fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.is_empty() || left.len() != right.len() {
        return None;
    }
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    let left_norm = left.iter().map(|v| v * v).sum::<f32>().sqrt();
    let right_norm = right.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !left_norm.is_finite()
        || !right_norm.is_finite()
        || left_norm <= f32::EPSILON
        || right_norm <= f32::EPSILON
    {
        return None;
    }
    Some((dot / (left_norm * right_norm)).clamp(-1.0, 1.0))
}

pub fn angle_rad(left: &[f32], right: &[f32]) -> Option<f32> {
    cosine_similarity(left, right).map(f32::acos)
}

pub fn angular_velocity_rad_per_wave(
    left: &[f32],
    right: &[f32],
    wave_delta: usize,
) -> Option<f32> {
    if wave_delta == 0 {
        return None;
    }
    angle_rad(left, right).map(|angle| angle / wave_delta as f32)
}

pub fn angular_novelty(left: &[f32], right: &[f32]) -> Option<f32> {
    angle_rad(left, right).map(|angle| angle / std::f32::consts::PI)
}

pub fn normalize_fragment(text: &str) -> String {
    let mut value = text.to_string();
    for marker in ["<end_of_turn>", "<eos>", "<|eot_id|>"] {
        value = value.replace(marker, " ");
    }
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if value.chars().filter(|c| c.is_alphanumeric()).count() == 0 || value.chars().count() <= 1 {
        String::new()
    } else {
        value
    }
}

pub fn segment_waves(text: &str, max_chars: usize) -> Vec<String> {
    let limit = max_chars.max(1);
    let mut waves = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate_len = current.len() + usize::from(!current.is_empty()) + word.len();
        if !current.is_empty() && candidate_len > limit {
            waves.push(current);
            current = String::new();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
        if word.ends_with(['.', '!', '?', '。', '！', '？']) {
            let normalized = normalize_fragment(&current);
            if !normalized.is_empty() {
                waves.push(normalized);
            }
            current.clear();
        }
    }
    let normalized = normalize_fragment(&current);
    if !normalized.is_empty() {
        waves.push(normalized);
    }
    waves
}

fn trigrams(text: &str) -> HashSet<String> {
    let words: Vec<String> = normalize_fragment(text)
        .split_whitespace()
        .map(|word| word.to_lowercase())
        .collect();
    words.windows(3).map(|window| window.join(" ")).collect()
}

fn repetition(previous: &str, current: &str) -> f32 {
    let previous = trigrams(previous);
    let current = trigrams(current);
    if previous.is_empty() || current.is_empty() {
        return 0.0;
    }
    current.intersection(&previous).count() as f32 / current.len() as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagnationDecision {
    pub confirmed: bool,
    pub streak: usize,
    pub ignored: bool,
}

pub struct StagnationDetector {
    min_confirmations: usize,
    cosine_threshold: f32,
    angular_novelty_threshold: f32,
    repetition_threshold: f32,
    previous_text: Option<String>,
    previous_embedding: Vec<f32>,
    previous_tokens: usize,
    streak: usize,
}

impl Default for StagnationDetector {
    fn default() -> Self {
        Self::new(3)
    }
}

impl StagnationDetector {
    pub fn new(min_confirmations: usize) -> Self {
        Self {
            min_confirmations: min_confirmations.max(2),
            cosine_threshold: 0.92,
            angular_novelty_threshold: 0.08,
            repetition_threshold: 0.60,
            previous_text: None,
            previous_embedding: Vec::new(),
            previous_tokens: 0,
            streak: 0,
        }
    }

    pub fn observe(
        &mut self,
        text: &str,
        embedding: &[f32],
        token_count: usize,
    ) -> StagnationDecision {
        let text = normalize_fragment(text);
        if text.is_empty() || embedding.is_empty() {
            return StagnationDecision {
                confirmed: false,
                streak: self.streak,
                ignored: true,
            };
        }
        let mut stagnant = false;
        if let (Some(previous_text), Some(cosine), Some(novelty)) = (
            self.previous_text.as_deref(),
            cosine_similarity(&self.previous_embedding, embedding),
            angular_novelty(&self.previous_embedding, embedding),
        ) {
            stagnant = cosine >= self.cosine_threshold
                && novelty <= self.angular_novelty_threshold
                && repetition(previous_text, &text) >= self.repetition_threshold
                && token_count > self.previous_tokens;
        }
        self.previous_text = Some(text);
        self.previous_embedding = embedding.to_vec();
        self.previous_tokens = token_count;
        self.streak = if stagnant { self.streak + 1 } else { 0 };
        StagnationDecision {
            confirmed: self.streak >= self.min_confirmations,
            streak: self.streak,
            ignored: false,
        }
    }

    pub fn reset(&mut self) {
        self.previous_text = None;
        self.previous_embedding.clear();
        self.previous_tokens = 0;
        self.streak = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_is_bounded() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), Some(1.0));
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]).is_none());
        assert!(
            (angle_rad(&[1.0, 0.0], &[0.0, 1.0]).unwrap() - std::f32::consts::FRAC_PI_2).abs()
                < 1e-6
        );
    }

    #[test]
    fn detector_requires_three_confirmations_and_ignores_noise() {
        let mut detector = StagnationDetector::default();
        assert!(detector.observe("\n\n", &[1.0, 0.0], 1).ignored);
        for tokens in [10, 11, 12] {
            assert!(
                !detector
                    .observe("the same semantic three gram repeats", &[1.0, 0.0], tokens)
                    .confirmed
            );
        }
        assert!(
            detector
                .observe("the same semantic three gram repeats", &[1.0, 0.0], 13)
                .confirmed
        );
    }
}
