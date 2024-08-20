use anyhow::{Context, Error, Result};
use async_walkdir::{Filtering, WalkDir};
use clap::{Parser, Subcommand};
use spinoff::{spinners, Spinner};
use std::path::{Path, PathBuf};
use tokio::{
    fs::{copy, rename},
    sync::Semaphore,
    task::JoinSet,
};
use tokio_stream::StreamExt;

pub mod audio;
use audio::is_audio_file;

pub mod processing;
use processing::{get_result_path, process, Label, ResultPathOptions};

const MAX_OPEN_FILES: usize = 128;
static PERMITS: Semaphore = Semaphore::const_new(MAX_OPEN_FILES);

#[derive(Parser)]
#[command(about, long_about = None, version)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(required = true)]
    path: PathBuf,

    #[arg(short, long, default_value = "0.0.0.0:8000")]
    address: String,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    #[command()]
    Copy {
        #[arg(short, long)]
        speech_dir: Option<PathBuf>,

        #[arg(short, long)]
        music_dir: Option<PathBuf>,

        #[arg(short, long)]
        noise_dir: Option<PathBuf>,
    },

    #[command()]
    Move {
        #[arg(short, long)]
        speech_dir: Option<PathBuf>,

        #[arg(short, long)]
        music_dir: Option<PathBuf>,

        #[arg(short, long)]
        noise_dir: Option<PathBuf>,
    },
}

impl Command {
    async fn perform(&self, path: &Path, label: &Label, options: &ResultPathOptions) -> Result<()> {
        if let Some(dir) = get_result_path(path, label, options) {
            match self {
                Command::Copy { .. } => {
                    let _ = copy(path, &dir).await?;
                    return Ok(());
                }
                Command::Move { .. } => {
                    rename(path, &dir).await?;
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    let url = format!("http:/{}/", args.address);
    let options = match args.command {
        Some(Command::Copy {
            ref speech_dir,
            ref music_dir,
            ref noise_dir,
        }) => Some(ResultPathOptions {
            speech_dir: speech_dir.clone(),
            music_dir: music_dir.clone(),
            noise_dir: noise_dir.clone(),
        }),
        Some(Command::Move {
            ref speech_dir,
            ref music_dir,
            ref noise_dir,
        }) => Some(ResultPathOptions {
            speech_dir: speech_dir.clone(),
            music_dir: music_dir.clone(),
            noise_dir: noise_dir.clone(),
        }),
        _ => None,
    };

    let mut spinner = Spinner::new(spinners::Line, "Loading...", None);

    if args.path.is_file() {
        let permit = PERMITS
            .acquire()
            .await
            .with_context(|| "Failed to acquire permit")?;
        spinner.update_text(format!("Labelling {:?}", args.path));
        let (path, label) = process(args.path.clone(), url.clone(), permit)
            .await
            .with_context(|| format!("Failed to process {:?}", args.path))?;
        if let Some(ref command) = args.command {
            command
                .perform(&path, &label, &options.unwrap())
                .await
                .with_context(|| format!("failed to perform command {:?}", command))?;
        } else {
            spinner.stop_with_message(&format!("{:?}: {:?}", path, label));
        }
        return Ok(());
    }

    let mut entries = WalkDir::new(args.path).filter(|entry| async move {
        if is_audio_file(entry).await.unwrap_or(false) {
            Filtering::Continue
        } else {
            Filtering::Ignore
        }
    });
    let mut jobs = JoinSet::new();
    loop {
        match entries.next().await {
            Some(Ok(entry)) => {
                let permit = PERMITS
                    .acquire()
                    .await
                    .with_context(|| "Failed to acquire permit")?;
                let path = entry.path();
                spinner.update_text(format!("Labelling {:?}", path));
                let future = process(path, url.clone(), permit);
                jobs.spawn(future);
            }
            Some(Err(e)) => return Err(e.into()),
            None => break,
        };
    }
    spinner.stop();
    while let Some(result) = jobs.join_next().await {
        let (path, label) = result?.with_context(|| "Failed to label a file")?;
        println!("{:?}: {:?}", path, label);
        if let Some(ref command) = args.command {
            command
                .perform(&path, &label, &options.clone().unwrap())
                .await
                .with_context(|| format!("failed to perform command {:?}", command))?;
        }
    }
    Ok(())
}
