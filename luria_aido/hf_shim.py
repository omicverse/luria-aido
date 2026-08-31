"""让本机的 transformers 4.50.3 能加载这几个模型。两处补丁，都不改环境。

1) wandb 屏蔽。`wandb.proto.wandb_internal_pb2` 和已装的 protobuf 版本不匹配
   （`_MailboxSlot._result: Optional[pb.Result]` 找不到 Result），而
   `transformers.models.timm_wrapper` 的导入链会碰到它。任何走 AutoModel /
   EncoderDecoderModel 的加载都要遍历全量模型映射，于是全部被这一个坏包带崩。
   `sys.modules['wandb'] = None` 让 `import wandb` 抛 ImportError，
   transformers 的 `is_wandb_available()` 因此返回 False 并跳过 —— 这正是
   它对"没装 wandb"的正常处理路径。修 wandb 本身会动到别的东西，不值。

2) masking_utils 补丁。MolFormer 的 remote code 用了 4.5x+ 才有的
   `create_bidirectional_mask`，只用一次，做的就是把 (B,L) 的 0/1 mask 扩成
   加性 mask。补一个同名模块，而不是升级 transformers —— 4.50.3 正撑着
   Geneformer / ESM2 / ESMFold / ChemBERTa。
"""
import sys, types

sys.modules.setdefault("wandb", None)

import torch  # noqa: E402
import transformers  # noqa: E402

if not hasattr(transformers, "masking_utils"):
    m = types.ModuleType("transformers.masking_utils")

    def create_bidirectional_mask(config=None, inputs_embeds=None,
                                  attention_mask=None, **kw):
        dtype = inputs_embeds.dtype
        if attention_mask is None:
            b, l = inputs_embeds.shape[:2]
            attention_mask = torch.ones((b, l), device=inputs_embeds.device)
        ext = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    m.create_bidirectional_mask = create_bidirectional_mask
    sys.modules["transformers.masking_utils"] = m
    transformers.masking_utils = m
