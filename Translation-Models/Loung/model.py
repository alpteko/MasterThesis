import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.SequenceRecordInputter(),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=256),
      encoder=onmt.encoders.UnidirectionalRNNEncoder(
          num_layers=4,
          num_units=512,
          cell_class=tf.contrib.rnn.GRUCell,
          dropout=0.2,
          residual_connections=True),
      decoder=onmt.decoders.AttentionalRNNDecoder(
          num_layers=4,
          num_units=512,
          bridge=onmt.layers.CopyBridge(),
          attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
          cell_class=tf.contrib.rnn.GRUCell,
          dropout=0.2,
residual_connections=True))
