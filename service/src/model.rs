use anyhow::Result;
use itertools::Itertools;
use serde::Serialize;
use std::path::Path;
use tch::{autocast, no_grad, CModule, Device, Tensor};

#[derive(Debug, Clone, Copy, Serialize)]
pub enum Label {
    Speech,
    Music,
    Noise,
}

pub struct Model {
    model: CModule,
}

impl Model {
    pub fn new<T>(model_path: T) -> Result<Model>
    where
        T: AsRef<Path>,
    {
        let device = Device::cuda_if_available();
        println!("Running model on {:?}", device);
        Ok(Self {
            model: CModule::load_on_device(model_path, device)?,
        })
    }

    pub fn label(&self, tensor: &Tensor) -> Result<Box<[Label]>> {
        let output = no_grad(|| autocast(true, || tensor.apply(&self.model))).f_sigmoid()?;
        let labels = Vec::try_from(output.flatten(0, -1))?
            .chunks(527)
            .map(|chunk| {
                let output: [f32; 527] = chunk.try_into().unwrap();
                let results = [
                    (Label::Speech, output[0]),
                    (Label::Music, output[137]),
                    (Label::Noise, output[513]),
                ];
                let label = {
                    if results[0].1 < 0.5 && results[1].1 < 0.5 {
                        Label::Noise
                    } else {
                        results
                            .iter()
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .map(|(label, _)| label.to_owned())
                            .unwrap_or(Label::Noise)
                    }
                };
                label
            })
            .collect_vec()
            .into_boxed_slice();
        Ok(labels)
    }
}
