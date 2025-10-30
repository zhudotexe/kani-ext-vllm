from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import VLLMEngine
    from .http_engine import VLLMServerEngine
    from .openai_compat_engine import VLLMOpenAIEngine
else:
    # lazy-load the engines to avoid expensive imports
    def __getattr__(name):
        match name:
            case "VLLMEngine":
                from .engine import VLLMEngine

                return VLLMEngine
            case "VLLMServerEngine":
                from .http_engine import VLLMServerEngine

                return VLLMServerEngine
            case "VLLMOpenAIEngine":
                from .openai_compat_engine import VLLMOpenAIEngine

                return VLLMOpenAIEngine
        raise AttributeError(name=name)
