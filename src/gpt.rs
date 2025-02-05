use tch::{
    nn::{Embedding, Linear, Module, Sequential},
    Tensor,
};

use crate::attention::MultiHeadAttention;

#[derive(Clone, Copy)]
pub struct GPTConfig {
    pub vocab_size: i64,
    pub context_length: i64,
    pub emb_dim: i64,
    pub n_heads: i64,
    pub n_layers: i64,
    pub drop_rate: f64,
    pub qkv_bias: bool,
}

impl GPTConfig {
    const fn new(
        vocab_size: i64,
        context_length: i64,
        emb_dim: i64,
        n_heads: i64,
        n_layers: i64,
        drop_rate: f64,
        qkv_bias: bool,
    ) -> Self {
        Self {
            vocab_size,
            context_length,
            emb_dim,
            n_heads,
            n_layers,
            drop_rate,
            qkv_bias,
        }
    }
    pub const GPT2_124M: GPTConfig = GPTConfig::new(50_257, 1_024, 768, 12, 12, 0.1, false);
    pub const GPT2_MEDIUM: GPTConfig = GPTConfig::new(50_257, 1_024, 1_024, 16, 24, 0.1, false);
    pub const GPT2_LARGE: GPTConfig = GPTConfig::new(50_257, 1_024, 1_280, 20, 36, 0.1, false);
    pub const GPT2_XLARGE: GPTConfig = GPTConfig::new(50_257, 1_024, 1_600, 25, 48, 0.1, false);
}

#[derive(Debug)]
struct LayerNorm {
    eps: f64,
    scale: Tensor,
    shift: Tensor,
}

impl LayerNorm {
    fn new(vs: &tch::nn::Path, emb_dim: i64) -> Self {
        Self {
            eps: 1e-5,
            scale: vs.ones("scale", &[emb_dim]),
            shift: vs.zeros("shift", &[emb_dim]),
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mean = xs.mean_dim(-1, true, xs.kind());
        let var = xs.var_dim(-1, false, true);
        let norm_x = (xs - mean) / (var + self.eps).sqrt();
        self.scale.copy() * norm_x + self.shift.copy()
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
struct GELU;

impl Module for GELU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let root = Tensor::from(2. / std::f32::consts::PI).sqrt();
        let tanh: Tensor = root * (xs + 0.044715 * xs.pow_tensor_scalar(3));
        0.5 * xs * (1. + tanh.tanh())
    }
}
#[derive(Debug)]
struct FeedForward {
    layers: Sequential,
}

impl FeedForward {
    fn new(vs: &tch::nn::Path, cfg: GPTConfig) -> Self {
        let c = tch::nn::LinearConfig::default();
        let layers = tch::nn::seq()
            .add(tch::nn::linear(vs, cfg.emb_dim, 4 * cfg.emb_dim, c))
            .add(GELU)
            .add(tch::nn::linear(vs, 4 * cfg.emb_dim, cfg.emb_dim, c));
        Self { layers }
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.layers.forward(xs)
    }
}

#[derive(Debug)]
struct TransformerBlock {
    att: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_shortcut: f64,
}

impl TransformerBlock {
    fn new(vs: &tch::nn::Path, cfg: GPTConfig) -> Self {
        Self {
            att: MultiHeadAttention::new(
                vs,
                cfg.emb_dim,
                cfg.emb_dim,
                cfg.context_length,
                cfg.drop_rate,
                cfg.n_heads,
                cfg.qkv_bias,
            ),
            ff: FeedForward::new(vs, cfg),
            norm1: LayerNorm::new(vs, cfg.emb_dim),
            norm2: LayerNorm::new(vs, cfg.emb_dim),
            drop_shortcut: cfg.drop_rate,
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut shortcut = xs.copy();
        let mut x = self.norm1.forward(xs);
        x = self.att.forward(&x);
        x = x.dropout(self.drop_shortcut, true);
        x = &x + shortcut;

        shortcut = x.copy();
        x = self.norm2.forward(&x);
        x = self.ff.forward(&x);
        x = x.dropout(self.drop_shortcut, true);
        x = &x + shortcut;
        x
    }
}

#[derive(Debug)]
pub struct GPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: f64,
    trf_blocks: Sequential,
    final_norm: LayerNorm,
    out_head: Linear,
}

impl GPTModel {
    pub fn new(vs: &tch::nn::Path, cfg: GPTConfig) -> Self {
        let mut trf_blocks = tch::nn::seq();
        for _ in 0..cfg.n_layers {
            trf_blocks = trf_blocks.add(TransformerBlock::new(vs, cfg));
        }
        let config = tch::nn::EmbeddingConfig::default();
        let c = tch::nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            tok_emb: tch::nn::embedding(vs, cfg.vocab_size, cfg.emb_dim, config),
            pos_emb: tch::nn::embedding(vs, cfg.context_length, cfg.emb_dim, config),
            drop_emb: cfg.drop_rate,
            trf_blocks,
            final_norm: LayerNorm::new(vs, cfg.emb_dim),
            out_head: tch::nn::linear(vs, cfg.emb_dim, cfg.vocab_size, c),
        }
    }
}

impl Module for GPTModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let seq_len = xs.size()[1];
        let tok_embeds = self.tok_emb.forward(xs);

        let pos_embeds = self
            .pos_emb
            .forward(&Tensor::arange(seq_len, (xs.kind(), xs.device())));
        let mut x = tok_embeds + pos_embeds;
        x = x.dropout(self.drop_emb, true);
        x = self.trf_blocks.forward(&x);
        x = self.final_norm.forward(&x);
        let logits = self.out_head.forward(&x);
        #[allow(clippy::let_and_return)]
        logits
    }
}
