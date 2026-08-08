# ComfyUI-MiniMaxH3-FP16Safe
# 让 MiniMax H3 在 fp16 计算下数值稳定（V100 等无 bf16/fp8 硬件的显卡），
# 大矩阵乘保持 fp16 Tensor Core 速度。详见 README.md。
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
