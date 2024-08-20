use eyre::{bail, Context, Result};

const NUM_MEL_BINS: usize = 128;

pub fn compute_fbank(samples: &[f32]) -> Result<Vec<[f32; NUM_MEL_BINS]>> {
    if samples.is_empty() {
        bail!("The samples array is empty. No features to compute.")
    }

    let mut result = unsafe {
        knf_rs_sys::ComputeFbank(
            samples.as_ptr(),
            samples.len().try_into().context("samples len")?,
        )
    };

    let frames = unsafe {
        std::slice::from_raw_parts(
            result.frames,
            (result.num_frames * NUM_MEL_BINS as i32) as usize,
        )
        .to_vec()
    };

    let features = unsafe {
        Box::from_raw(frames.as_ptr() as *mut [[f32; NUM_MEL_BINS]; 0])
            .to_vec()
    };

    unsafe {
        knf_rs_sys::DestroyFbankResult(&mut result as *mut _);
    }

    Ok(features)
}

pub fn convert_integer_to_float_audio(samples: &[i16], output: &mut [f32]) {
    for (input, output) in samples.iter().zip(output.iter_mut()) {
        *output = *input as f32 / 32768.0;
    }
}

#[cfg(test)]
mod tests {
    use crate::compute_fbank;
    use std::f32::consts::PI;

    fn generate_sine_wave(sample_rate: usize, duration: usize, frequency: f32) -> Vec<f32> {
        let waveform_size = sample_rate * duration;
        let mut waveform = Vec::with_capacity(waveform_size);

        for i in 0..waveform_size {
            let sample = 0.5 * (2.0 * PI * frequency * i as f32 / sample_rate as f32).sin();
            waveform.push(sample);
        }
        waveform
    }

    #[test]
    fn it_works() {
        let sample_rate = 16000;
        let duration = 1; // 1 second
        let frequency = 440.0; // A4 note

        let waveform = generate_sine_wave(sample_rate, duration, frequency);
        let features = compute_fbank(&waveform);
        println!("features: {:?}", features);
    }
}
