use anyhow::Result;
use axum::{body::Bytes, http::StatusCode, routing::post, serve, Json, Router};
use clap::Parser;
use parse_duration::parse;
use queue::run;
use safetensors::SafeTensors;
use tensor::{fit, normalize, to_tensor};
use tokio::{net::TcpListener, spawn, task::spawn_blocking};

mod tensor;

mod model;
use model::{Label, Model};

mod queue;

#[derive(Debug, Parser)]
#[command(about, long_about = None, version)]
struct Args {
    #[arg(short, long, default_value = "0.0.0.0:8000")]
    address: String,

    #[arg(short, long, default_value = "./ast/model.pt")]
    model_path: String,

    #[arg(short, long, default_value_t = 1)]
    batch_size: usize,

    #[arg(short, long, default_value = "100ms")]
    timeout: String,
}

async fn handler(body: Bytes) -> Result<Json<Label>, (StatusCode, String)> {
    let tensor = spawn_blocking(move || {
        let tensors = SafeTensors::deserialize(&body[..])
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        let fbank = tensors
            .tensor("fbank")
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        let mut tensor = to_tensor(fbank)?;
        fit(&mut tensor)?;
        normalize(&mut tensor)?;
        Ok(tensor)
    })
    .await
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))??;

    let result_rx = queue::add(tensor).await;
    let label = result_rx
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(label))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let timeout = parse(&args.timeout)?;
    let model = Model::new(args.model_path)?;
    spawn(async move {
        run(model, args.batch_size, timeout).await;
    });

    let router = Router::new().route("/", post(handler));

    let listener = TcpListener::bind(args.address).await?;
    Ok(serve(listener, router).await?)
}
