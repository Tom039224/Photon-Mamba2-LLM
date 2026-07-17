//! `pm` CLI entry point.

use anyhow::Result;
use clap::Parser;

mod config;
mod eval_cmd;
mod generate_cmd;
mod model_build;
mod train_cmd;

/// Which compute backend to use for model operations.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
pub enum BackendKind {
    /// Candle-based backend (CPU or CUDA via Candle).
    #[default]
    Candle,
    /// Native CUDA backend (cudarc, requires `--features cuda`).
    #[cfg(feature = "cuda")]
    Cuda,
}

#[derive(Parser, Debug)]
#[command(name = "pm", version, about = "Photon x Mamba2 LLM CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Train a PhotonMamba model from a TOML config.
    Train(train_cmd::TrainArgs),
    /// Generate continuation token ids from a prompt.
    Generate(generate_cmd::GenerateArgs),
    /// Evaluation subcommands.
    #[command(subcommand)]
    Eval(EvalCommand),
}

#[derive(clap::Subcommand, Debug)]
enum EvalCommand {
    /// HellaSwag length-normalised log-prob scoring.
    Hellaswag(eval_cmd::EvalArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => train_cmd::run(args),
        Command::Generate(args) => generate_cmd::run(args),
        Command::Eval(EvalCommand::Hellaswag(args)) => eval_cmd::run(args),
    }
}
