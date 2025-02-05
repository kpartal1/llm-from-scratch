use tch::{
    nn::{Linear, Module},
    Tensor,
};

#[derive(Debug)]
pub struct MultiHeadAttention {
    d_out: i64,
    num_heads: i64,
    head_dim: i64,
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    out_proj: Linear,
    dropout: f64,
    mask: Tensor,
}

impl MultiHeadAttention {
    pub fn new(
        vs: &tch::nn::Path,
        d_in: i64,
        d_out: i64,
        context_length: i64,
        dropout: f64,
        num_heads: i64,
        qkv_bias: bool,
    ) -> Self {
        assert!(
            d_out % num_heads == 0,
            "d_out must be divisible by num_heads"
        );
        let c = tch::nn::LinearConfig {
            bias: qkv_bias,
            ..Default::default()
        };
        MultiHeadAttention {
            d_out,
            num_heads,
            head_dim: d_out / num_heads,
            w_query: tch::nn::linear(vs, d_in, d_out, c),
            w_key: tch::nn::linear(vs, d_in, d_out, c),
            w_value: tch::nn::linear(vs, d_in, d_out, c),
            out_proj: tch::nn::linear(vs, d_out, d_out, tch::nn::LinearConfig::default()),
            dropout,
            mask: vs.ones("mask", &[context_length, context_length]).triu(1),
        }
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let size = xs.size();
        let (b, num_tokens) = (size[0], size[1]);

        let size = [b, num_tokens, self.num_heads, self.head_dim];
        let keys = self.w_key.forward(xs).view(size).transpose(1, 2);
        let queries = self.w_query.forward(xs).view(size).transpose(1, 2);
        let values = self.w_value.forward(xs).view(size).transpose(1, 2);

        let mask_bool = self
            .mask
            .to_kind(tch::Kind::Bool)
            .narrow(0, 0, num_tokens)
            .narrow(1, 0, num_tokens);
        let attn_scores = queries
            .matmul(&keys.transpose(2, 3))
            .masked_fill(&mask_bool, f64::NEG_INFINITY);

        let attn_weights = (attn_scores / (*keys.size().last().unwrap() as f64).sqrt())
            .softmax(-1, tch::Kind::Float)
            .dropout(self.dropout, true);

        let context_vec = (attn_weights.matmul(&values))
            .transpose(1, 2)
            .contiguous()
            .view([b, num_tokens, self.d_out]);

        let context_vec = self.out_proj.forward(&context_vec);
        #[allow(clippy::let_and_return)]
        context_vec
    }
}
