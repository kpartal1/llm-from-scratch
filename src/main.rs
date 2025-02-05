use gpt::GPTConfig;
use std::fs;
use tch::nn::Module;
use tch::{IndexOp, Tensor};
mod attention;
mod data;
mod gpt;

fn main() {
    let device = tch::Device::cuda_if_available();
    println!("{device:?}");
    tch::manual_seed(123);
    // let raw_text =
    //     fs::read_to_string("resources/the-verdict.txt").expect("Could not open the file to read.");

    let tokenizer = tiktoken_rs::get_bpe_from_model("gpt-2").unwrap();
    // let vocab_size = 50257;
    // let output_dim = 256;
    let vs = tch::nn::VarStore::new(device);
    let root = vs.root();

    let model = gpt::GPTModel::new(&root, GPTConfig::GPT2_XLARGE);

    let start_context = "Hello, I am";
    let encoded = tokenizer
        .encode_ordinary(start_context)
        .into_iter()
        .map(|rank| rank as i32)
        .collect::<Vec<_>>();
    println!("encoded: {encoded:?}");
    let encoded_tensor = Tensor::from_slice(&encoded).to_device(device).unsqueeze(0);
    println!("type: {:?}", encoded_tensor.kind());
    println!("encoded_tensor.shape: {:?}", encoded_tensor.size());

    let out = tch::no_grad(|| {
        generate_text_simple(
            &model,
            encoded_tensor,
            6,
            GPTConfig::GPT2_124M.context_length,
        )
    });
    println!("Output: {out}");
    println!("Output length: {}", out.get(0).size()[0]);

    let decoded_text = tokenizer
        .decode(
            out.squeeze()
                .iter::<i64>()
                .unwrap()
                .map(|i| i as u32)
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    println!("{decoded_text}");
}

fn generate_text_simple(
    model: &impl Module,
    idx: Tensor,
    max_new_tokens: i64,
    context_size: i64,
) -> Tensor {
    let mut idx = idx;
    for _ in 0..max_new_tokens {
        let seq_len = idx.size()[1];
        let start_token_index = 0.max(seq_len - context_size);
        let idx_cond = idx.i((.., start_token_index..));
        let logits = tch::no_grad(|| model.forward(&idx_cond));
        let logits = logits.i((.., -1, ..));
        let probas = logits.softmax(-1, logits.kind());

        let idx_next = probas.argmax(-1, true);
        idx = Tensor::cat(&[idx, idx_next], 1);
    }
    idx
}

fn print_gradients(vs: &tch::nn::VarStore, model: &impl Module, x: &Tensor) {
    let output = model.forward(x);
    let target = Tensor::from_slice2(&[[0.]])
        .to_kind(tch::Kind::Float)
        .to_device(vs.device());

    let loss = output.mse_loss(&target, tch::Reduction::Mean);

    loss.backward();

    let mut variables = vs
        .variables()
        .into_iter()
        .filter(|(s, p)| s.contains("weight") && p.grad().defined())
        .collect::<Vec<_>>();
    variables.sort_unstable_by(|(s1, _), (s2, _)| s1.cmp(s2));

    for (name, param) in variables {
        println!(
            "{name} has gradient mean of {}",
            param.grad().abs().mean(tch::Kind::Float)
        );
    }
}
