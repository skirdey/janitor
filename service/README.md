# Janitor Service

Runs the labelling.

## setup

### model

In order to build the model, you must have `conda` installed. Then, run `build_model.sh` to generate it.

### building

Run `cargo build --release` to compile for the CPU. If you have cuda, prepend the build command with `TORCH_CUDA_VERSION=cu118|cu121` depending on your version.

### running

You can see all the arguments with `cargo run -- --help`.
