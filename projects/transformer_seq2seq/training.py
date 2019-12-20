from parlai.scripts.train_model import TrainLoop
from parlai.scripts.train_model import setup_args as tm_setupargs

def setup_args():
    """Defaults for baseline model"""
    parser = tm_setupargs()

    parser.set_defaults(
        model='projects.transformer_seq2seq.no_hugging.agents:TransformerGeneratorAgent',  # noqa: E501
        dict_lower=True,
        dict_tokenizer='bpe',
        batchsize=64,
        embeddingsize=768,
        learningrate=5e-4,
        warmup_updates=100,
        lr_scheduler_patience=0,
        lr_scheduler_decay=0.4,
        dropout=0.1,
        gradient_clip=0.1,
        optimizer='adamax',
        context_length=-1,
        person_tokens=True,
        add_p1_after_newln=True,
        inference='beam',
        beam_size=3,
        beam_min_n_best=30,
        variant='xlm',
        learn_positional_embeddings=True,
        n_layers=12,
        n_heads=12,
        ffn_size=3072,
        attention_dropout=0.1,
        relu_dropout=0.0,
        n_positions=1024,
        activation='gelu',
        embeddings_scale=False,
        n_segments=2,
        # validation_every_n_secs=300,
        validation_every_epochs=0.5,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=12,
        log_every_n_secs=20,
        load_from_checkpoint=False,
        save_after_valid = True,
        tensorboard_log=True,
        tensorboard_tag='task,batchsize,hiddensize,embeddingsize,attention,numlayers,rnn_class,learningrate,dropout,gradient_clip',
        tensorboard_metrics='ppl,loss,accuracy',
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    print('-------------------------training')
    TrainLoop(opt).train()
    # python train_jp_seq2seq.py -mf \installation\~\ParlAI\data\models\jp_dialogue\jp_twitter_base
    # --dict-class projects.jp_dialogue.jp_retrieval.retrieval_agents:UniBiDictionaryAgent

    # finetune:
    # python train_jp_seq2seq.py -mf \installation\~\ParlAI\data\models\rachel\rachel
    # --init-model models:jp_dialogue/jp_twitter_base_bpe --dict_file models:jp_dialogue/jp_twitter_base_bpe
    # -bs 16 --dict-tokenizer bpe --train-folder rachel



    """
    python -u examples\train_model.py --init-model zoo:pretrained_transformers/bi_model_huge_reddit/model 
    --batchsize 16 -pyt convai2 --shuffle true --model projects.transformer_seq2seq.no_hugging.agents:TransformerGeneratorAgent 
    --eval-batchsize 6 --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 -lr 5e-05 --history-size 20 
    --label-truncate 72 --text-truncate 360 --num-epochs 10.0 --max_train_time 200000 -veps 0.5 -vme 8000 
    --validation-metric ppl --validation-metric-mode min --save-after-valid True --log_every_n_secs 20 --dict-tokenizer bpe 
    --dict-lower True --optimizer adamax --output-scaling 0.06 --variant xlm --learn-positional-embeddings True --n-layers 12 
    --n-heads 12 --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 
    --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 --share-word-embeddings False 
    --dict-endtoken __start__ --inference beam --beam-size 3 -mf \research\ParlAI_seq2seq\data\models\conv_seq2seq\model
    """