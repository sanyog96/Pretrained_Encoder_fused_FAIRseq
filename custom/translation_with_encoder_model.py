from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from dictionary_with_bert import DictionaryWithBert

@register_task('translation_with_pretrained_encoder_model')
class TranslationWithEncoderModelTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        parser.add_argument('--model', type=str, metavar='DIR', required=True,
                            help='path to the model')
        parser.add_argument('--fine-tuning', action='store_true',
                            help='if set, the encoder model will be tuned')
        parser.add_argument('--model-name', type=str, required=True,
                            help='name of the model used to pick the necessary library')
        parser.set_defaults(left_pad_source=False)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        if hasattr(args, 'model_overrides'):
            model_overrides = eval(args.model_overrides)
            model_overrides['model'] = args.model
            model_overrides['model_name'] = args.model_name
            args.model_overrides = "{}".format(model_overrides)

    @classmethod
    def setup_task(cls, args, **kwargs):
        task = super().setup_task(args, **kwargs)
        task.src_dict = DictionaryWithBert(task.src_dict)
        return task
