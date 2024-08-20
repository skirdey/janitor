use crate::audio::{create_fbank, extract_audio, resample, NUM_MEL_BINS, SAMPLE_RATE};
use anyhow::{Context, Result};
use byte_slice_cast::AsByteSlice;
use reqwest::Client;
use safetensors::{serialize, tensor::TensorView, Dtype};
use serde::Deserialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
use tokio::{
    fs::File,
    io::AsyncReadExt,
    sync::SemaphorePermit,
    task::{block_in_place, spawn_blocking},
};

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum Label {
    Speech,
    Music,
    Noise,
}

async fn label<'a>(tensor: &'a TensorView<'a>, url: String) -> Result<Label> {
    let bytes = block_in_place(|| -> Result<Vec<u8>> {
        let tensors = HashMap::from([("fbank", tensor)]);
        let bytes = serialize(tensors, &None).with_context(|| "Failed to serialize tensor")?;
        Ok(bytes)
    })?;

    let response = Client::new()
        .post(url.clone())
        .body(bytes)
        .send()
        .await
        .with_context(|| format!("Failed to send bytes to {}", url))?;
    let text = response
        .text()
        .await
        .with_context(|| "Failed to extract response body")?;
    let label = serde_json::from_str(&text).with_context(|| "Failed to deserialize output")?;
    Ok(label)
}

pub async fn process(
    path: PathBuf,
    url: String,
    _permit: SemaphorePermit<'_>,
) -> Result<(PathBuf, Label)> {
    let name = path
        .file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .to_string();
    let mut file = File::open(&path)
        .await
        .with_context(|| format!("Failed to open {}", name))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .await
        .with_context(|| format!("Failed to read file: {}", name))?;

    let fbank = spawn_blocking(move || -> Result<Box<[[f32; NUM_MEL_BINS]]>> {
        let mut audio = extract_audio(buffer).with_context(|| "Failed to extract audio")?;
        resample(&mut audio, SAMPLE_RATE);
        let fbank = create_fbank(&mut audio).with_context(|| "Failed to create filter bank")?;
        Ok(fbank)
    })
    .await??;
    let size = vec![fbank.len(), NUM_MEL_BINS];
    let tensor = TensorView::new(Dtype::F32, size, fbank.as_byte_slice())
        .with_context(|| "Failed to create tensor from fbank")?;

    let label = label(&tensor, url)
        .await
        .with_context(|| format!("Failed to label {}", name))?;
    Ok((path, label))
}

#[derive(Clone)]
pub struct ResultPathOptions {
    pub speech_dir: Option<PathBuf>,
    pub music_dir: Option<PathBuf>,
    pub noise_dir: Option<PathBuf>,
}

pub fn get_result_path(path: &Path, label: &Label, options: &ResultPathOptions) -> Option<PathBuf> {
    let mut dir = match label {
        Label::Speech => {
            if let Some(ref dir) = options.speech_dir {
                dir.clone()
            } else {
                return None;
            }
        }
        Label::Music => {
            if let Some(ref dir) = options.music_dir {
                dir.clone()
            } else {
                return None;
            }
        }
        Label::Noise => {
            if let Some(ref dir) = options.noise_dir {
                dir.clone()
            } else {
                return None;
            }
        }
    };
    let name = path
        .file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .to_string();
    dir.push(name);
    Some(dir)
}
