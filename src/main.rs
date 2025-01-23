use ai_dataloader::collate::TorchCollate;
use ai_dataloader::iterable::DataLoader;
use ai_dataloader::{Dataset, GetSample, Len};
use std::fs;
use tch::nn::Module;
use tch::Tensor;
use tiktoken_rs::CoreBPE;

fn main() {
    tch::manual_seed(123);
    let raw_text =
        fs::read_to_string("resources/the-verdict.txt").expect("Could not open the file to read.");

    let vocab_size = 50257;
    let output_dim = 256;
    let device = tch::Device::Cpu;
    let vs = tch::nn::VarStore::new(device);
    let token_embedding_layer = tch::nn::embedding(
        vs.root(),
        vocab_size,
        output_dim,
        tch::nn::EmbeddingConfig::default(),
    );

    let max_length = 4;
    let dataloader = create_dataloader_v1(&raw_text, 8, max_length, max_length);
    let mut data_iter = dataloader.into_iter();
    let (inputs, targets) = data_iter.next().unwrap();
    let inputs = Tensor::stack(&inputs, -1);
    let targets = Tensor::stack(&targets, -1);
    println!("Inputs:\n{inputs}\nTargets:\n{targets}");

    println!("Token IDs:\n{inputs}");
    println!("\nInputs shape:\n{}", inputs.internal_shape_as_tensor());
    let token_embeddings = token_embedding_layer.forward(&inputs);
    println!("{}", token_embeddings.internal_shape_as_tensor());

    let context_length = max_length as i64;
    let pos_embedding_layer = tch::nn::embedding(
        vs.root(),
        context_length,
        output_dim,
        tch::nn::EmbeddingConfig::default(),
    );
    let pos_embeddings =
        pos_embedding_layer.forward(&Tensor::arange(context_length, (tch::Kind::Int, device)));
    println!("{}", pos_embeddings.internal_shape_as_tensor());
    let input_embeddings = token_embeddings + pos_embeddings;
    println!("{}", input_embeddings.internal_shape_as_tensor());
}

struct GPTDatasetV1 {
    input_ids: Vec<Vec<i32>>,
    target_ids: Vec<Vec<i32>>,
}

impl Len for GPTDatasetV1 {
    fn len(&self) -> usize {
        self.input_ids.len()
    }
}

impl GetSample for GPTDatasetV1 {
    type Sample = (Vec<i32>, Vec<i32>);
    fn get_sample(&self, index: usize) -> Self::Sample {
        (
            self.input_ids[index].clone(),
            self.target_ids[index].clone(),
        )
    }
}

impl Dataset for GPTDatasetV1 {}

struct GPTDatasetV1Iter {
    dataset: GPTDatasetV1,
    index: usize,
}

impl Iterator for GPTDatasetV1Iter {
    type Item = (Vec<i32>, Vec<i32>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            None
        } else {
            let sample = self.dataset.get_sample(self.index);
            self.index += 1;
            Some(sample)
        }
    }
}

impl IntoIterator for GPTDatasetV1 {
    type IntoIter = GPTDatasetV1Iter;
    type Item = (Vec<i32>, Vec<i32>);
    fn into_iter(self) -> Self::IntoIter {
        GPTDatasetV1Iter {
            dataset: self,
            index: 0,
        }
    }
}

impl GPTDatasetV1 {
    pub fn new(txt: &str, tokenizer: CoreBPE, max_length: usize, stride: usize) -> Self {
        let mut input_ids = Vec::new();
        let mut target_ids = Vec::new();
        let token_ids = tokenizer
            .encode_ordinary(txt)
            .into_iter()
            .map(|item| item as i32)
            .collect::<Vec<_>>();
        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let input_chunk = &token_ids[i..i + max_length];
            let target_chunk = &token_ids[i + 1..i + max_length + 1];
            input_ids.push(Vec::from(input_chunk));
            target_ids.push(Vec::from(target_chunk));
        }
        Self {
            input_ids,
            target_ids,
        }
    }
}

fn create_dataloader_v1(
    txt: &str,
    batch_size: usize,
    max_length: usize,
    stride: usize,
) -> DataLoader<GPTDatasetV1, TorchCollate> {
    let tokenizer = tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
    let dataset = GPTDatasetV1::new(txt, tokenizer, max_length, stride);
    DataLoader::builder(dataset)
        .collate_fn(TorchCollate)
        .batch_size(batch_size)
        .drop_last()
        .build()
}
