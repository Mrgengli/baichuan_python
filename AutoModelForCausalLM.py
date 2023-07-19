class AutoModelForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForCausalLM is designed to be instantiated "
            "using the `AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForCausalLM.from_config(config)` methods."
        )

    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_CAUSAL_LM_MAPPING, use_model_types=False)
    def from_config(cls, config):

        if type(config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
            return MODEL_FOR_CAUSAL_LM_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_CAUSAL_LM_MAPPING.keys())
            )
        )


    @classmethod
    @replace_list_option_in_docstrings(MODEL_FOR_CAUSAL_LM_MAPPING)
    @add_start_docstrings(
        "Instantiate one of the model classes of the library---with a causal language modeling head---from a "
        "pretrained model.",
        AUTO_MODEL_PRETRAINED_DOCSTRING,
    )
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
            return MODEL_FOR_CAUSAL_LM_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__, cls.__name__, ", ".join(c.__name__ for c in MODEL_FOR_CAUSAL_LM_MAPPING.keys())
            )
        )
