# MasterThesis
Neural Sign Language By Learning Tokenization

# Tokenization
## CNN Training
They produce CNN models.
## CNN Feature Extraction 
They produce tfrecord and numpy features of the translation dataset.
Their form can be right-hand, full frame etc.
# NMT Module
It uses OpenNMT-tf library.
There are four moddels: Bahdanau, Loung, Transformer, Multi-Source
## Training a NMT Module
onmt train_and_eval --model_type [PythonModelFilePath] --config [ConfigYAMLFilePath]
## Creating Test and Eval Translation Sentences
### Average Models 
onmt-average-checkpoints \
    --model_dir [ModelDirectory] \
    --output_dir [OutputDirectory] \
    --max_count 10
### Infer Sentences
onmt-main infer \
    --config conf.yml \
    --features_file [TestFeatureFilePath] \
    --predictions_file [OutputPredictonFilePath] \
    --checkpoint_path [ModelPath(AverageModel)]
## Evaluation
Use eval scripts evaluation_utils and pbs
Code: evaluation_utils.evaluate(ref_file=,tran_file=, metric=)
Script: python3 pbs.py -reference-file REFERENCE_FILE --baseline-file
              BASELINE_FILE --sample-size SAMPLE_SIZE