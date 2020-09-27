import tensorflow as tf
import opennmt as onmt

class Transformer(onmt.models.Transformer):

  def __init__(self):
    super(Transformer, self).__init__(
      source_inputter=onmt.inputters.SequenceRecordInputter(),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=128),
      num_layers=6,
      num_units=256,
      num_heads=8,
      ffn_inner_dim=1024,
      dropout=0.1,
      attention_dropout=0.1,
      relu_dropout=0.1,
      share_encoders=False)

model = DualSourceTransformer
