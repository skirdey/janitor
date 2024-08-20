use anyhow::{Context, Error, Result};
use async_walkdir::DirEntry;
use fon::{chan::Ch32, Audio};
use itertools::Itertools;
use knf_rs::compute_fbank;
use rodio::{Decoder, Source};
use std::io::Cursor;

pub const SAMPLE_RATE: u32 = 16000;
pub const NUM_FRAMES: usize = 1024;
pub const NUM_MEL_BINS: usize = 128;

pub async fn is_audio_file(entry: DirEntry) -> Result<bool> {
    if let Some(mime) = mime_guess::from_path(entry.path()).first() {
        if mime.type_() != "audio" {
            return Ok(false);
        }
    }
    Ok(true)
}

fn extract_samples(decoder: Decoder<Cursor<Vec<u8>>>) -> Result<(Box<[f32]>, u32)> {
    let channels = decoder.channels() as usize;
    let sample_rate = decoder.sample_rate();
    let samples = decoder
        .map(|sample| sample as f32 / i16::MAX as f32)
        .chunks(channels)
        .into_iter()
        .map(|chunk| chunk.into_iter().sum::<f32>() / channels as f32)
        .collect_vec()
        .into_boxed_slice();
    Ok((samples, sample_rate))
}

pub fn extract_audio(buffer: Vec<u8>) -> Result<Audio<Ch32, 1>> {
    let cursor = Cursor::new(buffer);
    let decoder = Decoder::new(cursor).with_context(|| "Failed to initialize decoder")?;
    let (samples, sample_rate) =
        extract_samples(decoder).with_context(|| "Failed to extract samples")?;
    let audio = Audio::with_f32_buffer(sample_rate, samples);
    Ok(audio)
}

pub fn resample(audio: &mut Audio<Ch32, 1>, target_sample_rate: u32) {
    *audio = Audio::with_audio(target_sample_rate, audio);
}

pub fn create_fbank(audio: &mut Audio<Ch32, 1>) -> Result<Box<[[f32; NUM_MEL_BINS]]>> {
    let samples = audio.as_f32_slice();
    let mut fbank = compute_fbank(samples).map_err(|e| Error::msg(e.to_string()))?;
    fbank.truncate(NUM_FRAMES);
    Ok(fbank.into_boxed_slice())
}
