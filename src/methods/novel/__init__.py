"""Novel PEFT methods — import all to trigger @register_method decorators."""

from src.methods.novel.polyattn import PolyAttnMethod  # noqa: F401
from src.methods.novel.polyssf import PolySSFMethod  # noqa: F401
from src.methods.novel.polycomm import PolyCommMethod  # noqa: F401
from src.methods.novel.polyscale import PolyScaleMethod  # noqa: F401
from src.methods.novel.polycls import PolyCLSMethod  # noqa: F401
from src.methods.novel.attn_routing import AttnRoutingMethod  # noqa: F401
from src.methods.novel.head_interaction import HeadInteractionMethod  # noqa: F401
from src.methods.novel.spectral_attn import SpectralAttnMethod  # noqa: F401
from src.methods.novel.attn_normalizer import AttnNormalizerMethod  # noqa: F401
from src.methods.novel.rope_geometry import RoPEGeometryMethod  # noqa: F401
