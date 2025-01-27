use ai_dataloader::{
    collate::{Collate, DefaultCollate},
    iterable::DataLoader,
    Dataset, GetSample, Len,
};
use tch::Tensor;

pub struct Data {
    inputs: Tensor,
    targets: Tensor,
}

impl Collate<Data> for DefaultCollate {
    type Output = (Tensor, Tensor);
    fn collate(&self, batch: Vec<Data>) -> Self::Output {
        let inputs = Tensor::stack(
            &batch.iter().map(|d| d.inputs.copy()).collect::<Vec<_>>(),
            0,
        );
        let targets = Tensor::stack(
            &batch.iter().map(|d| d.targets.copy()).collect::<Vec<_>>(),
            0,
        );
        (inputs, targets)
    }
}

pub struct GPTDatasetV1 {
    input_ids: Vec<Tensor>,
    target_ids: Vec<Tensor>,
}

impl Len for GPTDatasetV1 {
    fn len(&self) -> usize {
        self.input_ids.len()
    }
}

impl GetSample for GPTDatasetV1 {
    type Sample = Data;
    fn get_sample(&self, index: usize) -> Self::Sample {
        Data {
            inputs: self.input_ids[index].copy(),
            targets: self.target_ids[index].copy(),
        }
    }
}

impl Dataset for GPTDatasetV1 {}

pub struct GPTDatasetV1Iter {
    dataset: GPTDatasetV1,
}

impl Iterator for GPTDatasetV1Iter {
    type Item = Data;
    fn next(&mut self) -> Option<Self::Item> {
        let inputs = self.dataset.input_ids.pop();
        let targets = self.dataset.target_ids.pop();
        match (inputs, targets) {
            (Some(inputs), Some(targets)) => Some(Data { inputs, targets }),
            _ => None,
        }
    }
}

impl IntoIterator for GPTDatasetV1 {
    type IntoIter = GPTDatasetV1Iter;
    type Item = Data;
    fn into_iter(self) -> Self::IntoIter {
        GPTDatasetV1Iter { dataset: self }
    }
}

impl GPTDatasetV1 {
    pub fn new(
        txt: &str,
        tokenizer: tiktoken_rs::CoreBPE,
        max_length: usize,
        stride: usize,
    ) -> Self {
        let token_ids = tokenizer
            .encode_ordinary(txt)
            .into_iter()
            .map(|item| item as i32)
            .collect::<Vec<_>>();
        let cap = token_ids.len() - max_length / stride;
        let mut input_ids = Vec::with_capacity(cap);
        let mut target_ids = Vec::with_capacity(cap);
        for i in (0..token_ids.len() - max_length).step_by(stride) {
            let inputs = Tensor::from_slice(&token_ids[i..i + max_length]);
            let targets = Tensor::from_slice(&token_ids[i + 1..i + max_length + 1]);
            input_ids.push(inputs);
            target_ids.push(targets);
        }
        Self {
            input_ids,
            target_ids,
        }
    }
}

pub fn create_dataloader_v1(
    txt: &str,
    batch_size: usize,
    max_length: usize,
    stride: usize,
) -> DataLoader<GPTDatasetV1, DefaultCollate> {
    let tokenizer = tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
    let dataset = GPTDatasetV1::new(txt, tokenizer, max_length, stride);
    DataLoader::builder(dataset)
        .batch_size(batch_size)
        .drop_last()
        .build()
}
