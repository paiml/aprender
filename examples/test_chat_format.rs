//! Test Qwen2 with proper chat template
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)?;
    let vocab = mapped.model.vocabulary().expect("vocab");

    // ChatML format for Qwen2-Instruct:
    // <|im_start|>system
    // You are a helpful assistant.<|im_end|>
    // <|im_start|>user
    // What is 2+2?<|im_end|>
    // <|im_start|>assistant

    // Special tokens:
    // <|im_start|> = 151644
    // <|im_end|> = 151645
    // <|endoftext|> = 151643

    // Find token IDs for the prompt
    // Let's manually construct the token sequence for:
    // "system\nYou are a helpful assistant." after <|im_start|>
    // For now, let's try a simpler test - just tokenize the input manually

    // Token IDs (from llama.cpp tokenization):
    // <|im_start|> = 151644
    // "system" = 9125
    // "\n" = 198
    // "You" = 2610
    // " are" = 525
    // " a" = 264
    // " helpful" = 10950
    // " assistant" = 17847
    // "." = 13
    // <|im_end|> = 151645

    // Let's start simpler - just test with <|im_start|>assistant\n
    // to see if the model continues correctly

    let im_start = 151644u32;
    let im_end = 151645u32;
    let assistant = 77091u32; // "assistant"
    let newline = 198u32; // "\n"
    let _system = 9125u32;

    // Test: <|im_start|>assistant\n
    let tokens = vec![im_start, assistant, newline];
    println!(
        "Testing with: <|im_start|>assistant\\n (tokens: {:?})",
        tokens
    );

    let logits = model.forward(&tokens)?;

    println!("\nTop 10 predictions:");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    // Test with a math question in chat format
    // <|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n
    let user = 872u32; // "user"
    let what = 3838u32; // "What"
    let is_ = 374u32; // " is"
    let space = 220u32; // " "
    let two = 17u32; // "2"
    let plus = 10u32; // "+"
    let _eq = 28u32; // "="
    let qmark = 30u32; // "?"

    let chat_tokens = vec![
        im_start, user, newline, // <|im_start|>user\n
        what, is_, space, two, plus, two, qmark, // What is 2+2?
        im_end, newline, // <|im_end|>\n
        im_start, assistant, newline, // <|im_start|>assistant\n
    ];

    println!("\n\nTesting chat format: <|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n");
    println!("Tokens: {:?}", chat_tokens);

    let logits = model.forward(&chat_tokens)?;

    println!("\nTop 10 predictions:");
    let mut indexed: Vec<_> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (tok, logit) in indexed.iter().take(10) {
        let tok_str = vocab.get(*tok).map_or("?", |s| s.as_str());
        println!("  Token {} ({:?}): logit={:.4}", tok, tok_str, logit);
    }

    // Check specific tokens
    println!("\nSpecific tokens of interest:");
    println!("  Token 19 (\"4\"): logit={:.4}", logits[19]);
    println!("  Token 0 (\"!\"): logit={:.4}", logits[0]);

    Ok(())
}
