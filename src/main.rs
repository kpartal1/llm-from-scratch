use attention::MultiHeadAttention;
use std::fs;
use tch::nn::Module;
use tch::Tensor;
mod attention;
mod data;

fn main() {
    tch::manual_seed(123);
    let raw_text =
        fs::read_to_string("resources/the-verdict.txt").expect("Could not open the file to read.");

    let vocab_size = 50257;
    let output_dim = 256;
    let device = tch::Device::Cuda(0);
    let vs = tch::nn::VarStore::new(device);
    let root = vs.root();
    let token_embedding_layer = tch::nn::embedding(
        &root,
        vocab_size,
        output_dim,
        tch::nn::EmbeddingConfig::default(),
    );

    let max_length = 4;
    let dataloader = data::create_dataloader_v1(&raw_text, 8, max_length, max_length);
    let mut data_iter = dataloader
        .into_iter()
        .map(|(inputs, targets)| (inputs.to_device(device), targets.to_device(device)));
    let (inputs, targets) = data_iter.next().unwrap();
    println!("Inputs:\n{inputs}\nTargets:\n{targets}");

    println!("Token IDs:\n{inputs}");
    println!("\nInputs shape:\n{:?}", inputs.size());
    let token_embeddings = token_embedding_layer.forward(&inputs);
    println!("{:?}", token_embeddings.size());

    let context_length = max_length as i64;
    let pos_embedding_layer = tch::nn::embedding(
        &root,
        context_length,
        output_dim,
        tch::nn::EmbeddingConfig::default(),
    );
    let pos_embeddings =
        pos_embedding_layer.forward(&Tensor::arange(context_length, (tch::Kind::Int, device)));
    println!("{:?}", pos_embeddings.size());
    let input_embeddings = token_embeddings + pos_embeddings;
    println!("{:?}", input_embeddings.size());

    let inputs = Tensor::from_slice2(&[
        &[0.43, 0.15, 0.89],
        &[0.55, 0.87, 0.66],
        &[0.57, 0.85, 0.64],
        &[0.22, 0.58, 0.33],
        &[0.77, 0.25, 0.10],
        &[0.05, 0.80, 0.55],
    ])
    .to_kind(tch::Kind::Float)
    .to_device(device);

    let batch = Tensor::stack(&[inputs.copy(), inputs], 0);
    println!("{:?}", batch.size());

    let size = batch.size();
    let (context_length, d_in) = (size[1], size[2]);
    let d_out = 2;
    let mha = MultiHeadAttention::new(&root, d_in, d_out, context_length, 0., 2, false);
    let context_vecs = mha.forward(&batch);
    println!("{context_vecs}");
    println!("context_vecs.shape: {:?}", context_vecs.size());
}
