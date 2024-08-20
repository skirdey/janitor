use lazy_static::lazy_static;
use spinoff::{spinners, Spinner};
use std::{cmp::min, collections::VecDeque};
use tch::Tensor;
use tokio::{
    sync::{
        oneshot::{channel, Receiver, Sender},
        Mutex,
    },
    time::{sleep, Duration, Instant},
};

use crate::model::{Label, Model};

type JobQueue = VecDeque<(Tensor, Sender<Label>)>;

lazy_static! {
    static ref QUEUE: Mutex<JobQueue> = Mutex::new(VecDeque::new());
}

pub async fn add(tensor: Tensor) -> Receiver<Label> {
    let (result_tx, result_rx) = channel();
    let job = (tensor, result_tx);
    QUEUE.lock().await.push_back(job);
    result_rx
}

async fn get_jobs(
    batch_size: usize,
    timeout: Duration,
) -> ((Vec<Tensor>, Vec<Sender<Label>>), usize) {
    let mut tensors = Vec::with_capacity(batch_size);
    let mut transmitters = Vec::with_capacity(batch_size);
    let mut remaining = batch_size;
    let mut rest = 0;
    let start = Instant::now();
    while remaining > 0 && Instant::now() - start < timeout {
        let mut queue = QUEUE.lock().await;
        let usable = min(remaining, queue.len());
        if usable > 0 {
            for _ in 0..usable {
                if let Some(job) = queue.pop_front() {
                    tensors.push(job.0);
                    transmitters.push(job.1);
                }
            }
            remaining -= usable;
            rest = queue.len();
        } else {
            drop(queue);
            sleep(Duration::from_millis(100)).await;
        }
    }
    ((tensors, transmitters), rest)
}

pub async fn run(model: Model, batch_size: usize, timeout: Duration) {
    let mut spinner = Spinner::new(spinners::Line, "Waiting for jobs", None);
    loop {
        let ((tensors, transmitters), remaining) = get_jobs(batch_size, timeout).await;
        if tensors.is_empty() {
            continue;
        }

        let tensor = match Tensor::f_stack(&tensors, 0) {
            Ok(tensor) => tensor,
            Err(_) => {
                continue;
            }
        };

        spinner.update_text(format!(
            "Executing {} job(s) with {} still remaining",
            tensors.len(),
            remaining
        ));
        let labels = match model.label(&tensor) {
            Ok(tensor) => tensor,
            Err(_) => {
                continue;
            }
        };

        for (&label, result_tx) in labels.iter().zip(transmitters.into_iter()) {
            _ = result_tx.send(label);
        }
        spinner.update_text("Waiting for jobs");
    }
}
