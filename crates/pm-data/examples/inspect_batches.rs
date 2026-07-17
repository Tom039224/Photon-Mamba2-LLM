//! Diagnostic: print decoded text of the first N batches from a real-text
//! training stream, exactly as the trainer would receive them.
//!
//! Used to investigate the 14h TinyStories loss plateau (2026-06-30):
//! verify the tokenizer round-trips, that `<|endoftext|>` literals are
//! NOT being char-encoded into garbage, and that successive batches
//! contain distinct content (i.e. the packer advances the stream).
//!
//! Usage:
//!   cargo run -p pm-data --example inspect_batches --release -- \
//!       data/tinystories_valid.txt tokenizer/gpt2.json 5

use pm_data::{PackedBatcher, TextFileSource};
use pm_tokenizer::BpeTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} <text_file> <tokenizer.json> <n_batches> [batch_size] [seq_len]",
            args[0]
        );
        std::process::exit(2);
    }
    let text_path = &args[1];
    let tokenizer_path = &args[2];
    let n_batches: usize = args[3].parse()?;
    let batch_size: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let seq_len: usize = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(512);

    let tk = BpeTokenizer::from_file(tokenizer_path)?;
    let mut source = TextFileSource::open(text_path, 50256)?;
    let batcher = PackedBatcher::new(batch_size, seq_len, 1, 50256)?;

    println!("=== tokenizer round-trip self-test ===");
    let sample_texts = [
        "Once upon a time, there was a little girl.",
        "<|endoftext|>",
        "Hello world. <|endoftext|> Next story.",
    ];
    for s in sample_texts {
        let ids = tk.encode(s, false)?;
        let decoded = tk.decode(&ids, false)?;
        println!("  in:  {:?}", s);
        println!("  ids: {:?}", ids);
        println!("  out: {:?}", decoded);
        println!();
    }

    println!("=== first {} batches from real stream ===", n_batches);
    let mut prev_first_id: Option<i64> = None;
    let mut prev_text: Option<String> = None;
    for b in 0..n_batches {
        let Some((ids, targets)) = batcher.next_batch(&mut source, &tk)? else {
            println!("batch {b}: stream exhausted");
            break;
        };
        let first_seq = &ids[..ids.len() / batch_size];
        let text = tk.decode(first_seq, false)?;
        let n_pad = first_seq.iter().filter(|&&id| id == 50256).count();
        let unique_ids: std::collections::HashSet<_> = first_seq.iter().collect();
        println!("--- batch {b} ---");
        println!(
            "  n_tokens = {}, unique = {}, n_separator(50256) = {}",
            first_seq.len(),
            unique_ids.len(),
            n_pad
        );
        println!(
            "  first 10 ids:   {:?}",
            &first_seq[..10.min(first_seq.len())]
        );
        println!("  first 10 tgts:  {:?}", &targets[..10.min(targets.len())]);
        let head: String = text.chars().take(200).collect();
        println!("  decoded head:   {:?}", head);
        let tail: String = text
            .chars()
            .rev()
            .take(100)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        println!("  decoded tail:   {:?}", tail);

        if let Some(prev_id) = prev_first_id {
            if prev_id == first_seq[0] && prev_text.as_deref() == Some(text.as_str()) {
                println!("  WARN: identical content to previous batch — packer not advancing!");
            }
        }
        prev_first_id = Some(first_seq[0]);
        prev_text = Some(text);
    }

    Ok(())
}
