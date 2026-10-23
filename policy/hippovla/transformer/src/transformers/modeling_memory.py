import math
import torch
import torch.nn as nn


class MemorySlotAttention(nn.Module):
    """Multi-head attention, handles [B, V, N, D] or [B, N, D] inputs."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm_q = nn.RMSNorm(dim)
        self.norm_kv = nn.RMSNorm(dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.dim() == 3:
            B, Nq, D = query.shape
            query = query.unsqueeze(1)
            key_value = key_value.unsqueeze(1)
            squeeze_back = True
        else:
            squeeze_back = False
            
        B, V, Nq, D = query.shape
        Nkv = key_value.shape[2]
        
        q = self.q_proj(self.norm_q(query.reshape(B * V * Nq, D))).reshape(B, V, Nq, D)
        k = self.k_proj(self.norm_kv(key_value.reshape(B * V * Nkv, D))).reshape(B, V, Nkv, D)
        v = self.v_proj(self.norm_kv(key_value.reshape(B * V * Nkv, D))).reshape(B, V, Nkv, D)
        
        q = q.view(B, V, Nq, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * V * self.num_heads, Nq, self.head_dim)
        k = k.view(B, V, Nkv, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * V * self.num_heads, Nkv, self.head_dim)
        v = v.view(B, V, Nkv, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * V * self.num_heads, Nkv, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, Nkv):
                raise ValueError(
                    f"key_padding_mask must be [B, Nkv], got {key_padding_mask.shape}"
                )
            valid = key_padding_mask[:, None, None, :].expand(B, V, self.num_heads, Nkv)
            valid = valid.reshape(B * V * self.num_heads, 1, Nkv)
            attn = attn.masked_fill(~valid, torch.finfo(attn.dtype).min)
            attn = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
            attn = attn * valid.to(attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            attn = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        
        out = (attn @ v)
        out = out.view(B, V, self.num_heads, Nq, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B, V, Nq, D)
        out = self.o_proj(out.reshape(B * V * Nq, D)).reshape(B, V, Nq, D)
        
        if squeeze_back:
            out = out.squeeze(1)
            
        return out


class ShortTermMemoryBank(nn.Module):
    """
    Short-term memory for multi-level (3 levels) dual-view features with batch dimension.
    预设时间步 T=5，3个层级独立处理，支持 batch 中不同 timestep。
    
    Input:
        - memory: [B, 3, T, 2, 64, D]，3个层级，每层级T步历史，双视角
        - visual: [B, 3, 2, 64, D]，3个层级，当前帧，双视角
        - timestep: [B] 或 int，每个样本的绝对时间步
        
    Output:
        - new_memory: [B, 3, 2, 64, D]，更新后的最新记忆状态（3个层级）
    """
    def __init__(
        self,
        dim: int,
        num_slots: int = 64,
        num_heads: int = 8,
        num_levels: int = 3,
        num_timesteps: int = 5,
        num_views: int = 2,
        use_slot_attention: bool = True,
        temporal_encoding: str = "absolute",
        fusion: str = "gated",
        injection_levels: tuple[int, ...] | list[int] | None = None,
        query_mode: str = "latest_history",
        memory_interval: int = 10,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_levels = num_levels
        self.num_timesteps = num_timesteps
        self.num_views = num_views
        self.use_slot_attention = use_slot_attention
        self.temporal_encoding = temporal_encoding
        self.fusion = fusion
        self.injection_levels = tuple(range(num_levels)) if injection_levels is None else tuple(injection_levels)
        self.query_mode = query_mode
        self.memory_interval = memory_interval

        if temporal_encoding not in {"absolute", "relative", "none"}:
            raise ValueError(f"Unsupported temporal_encoding: {temporal_encoding}")
        if fusion not in {"gated", "add"}:
            raise ValueError(f"Unsupported fusion: {fusion}")
        if query_mode not in {"latest_history", "current_visual", "mean_history"}:
            raise ValueError(f"Unsupported query_mode: {query_mode}")
        if not set(self.injection_levels).issubset(range(num_levels)):
            raise ValueError(
                f"injection_levels must be in [0, {num_levels - 1}], got {self.injection_levels}"
            )
        if memory_interval <= 0:
            raise ValueError(f"memory_interval must be positive, got {memory_interval}")
        
        self.write_attns = (
            nn.ModuleList([
                MemorySlotAttention(dim, num_heads) if level in self.injection_levels else nn.Identity()
                for level in range(num_levels)
            ])
            if use_slot_attention
            else None
        )
        self.write_gates = (
            nn.ModuleList([
                nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
                if level in self.injection_levels
                else nn.Identity()
                for level in range(num_levels)
            ])
            if fusion == "gated"
            else None
        )
        
        self.norms = nn.ModuleList([
            nn.RMSNorm(dim) if level in self.injection_levels else nn.Identity()
            for level in range(num_levels)
        ])
        
        self.register_buffer(
            "_temporal_div",
            torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        )
        
    def get_temporal_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        获取正弦时间编码，支持 batch。
        
        Args:
            timestep: [B] 或 scalar，每个样本的时间步
        
        Returns:
            emb: [B, D] 或 [D]
        """
        # 确保是 tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=torch.float32, device=self._temporal_div.device)
        # 处理 scalar 情况
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # [B, 1] * [D//2] -> [B, D//2]
        angles = timestep.unsqueeze(-1) * self._temporal_div.to(timestep.device)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, D]
        
        return emb
    
    def forward(
        self,
        memory: torch.Tensor,           # [B, 3, T, V, 64, D]
        visual: torch.Tensor,           # [B, 3, V, 64, D]
        timestep: torch.Tensor,         # [B] 或 int，每个样本的时间步
    ) -> torch.Tensor:
        """
        更新记忆，3个层级分别处理，支持 batch 中不同 timestep。
        """
        assert memory.dim() == 6, f"memory must be [B,3,T,V,64,D], got {memory.shape}"
        assert visual.dim() == 5, f"visual must be [B,3,V,64,D], got {visual.shape}"
        B, L, T, V, S, D = memory.shape
        assert L == self.num_levels
        assert T == self.num_timesteps
        # assert V == self.num_views
        assert S == self.num_slots
        assert visual.shape == (B, L, V, S, D)
        
        # 处理 timestep：统一为 [B]
        if isinstance(timestep, int):
            timestep = torch.full((B,), timestep, dtype=torch.float32, device=memory.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(B)
        elif timestep.shape[0] != B:
            raise ValueError(f"timestep batch size {timestep.shape[0]} != memory batch size {B}")
        
        timestep = timestep.to(device=memory.device, dtype=torch.float32)
        # Memory is ordered from oldest to newest: [t-T*I, ..., t-I].
        lags = torch.arange(T, 0, -1, device=memory.device, dtype=torch.float32) * self.memory_interval
        valid_steps = timestep[:, None] >= lags[None, :]

        if self.temporal_encoding == "absolute":
            history_steps = (timestep[:, None] - lags[None, :]).clamp_min(0)
            history_t_emb = self.get_temporal_embedding(history_steps.reshape(-1)).view(B, T, D)
            current_t_emb = self.get_temporal_embedding(timestep)
        elif self.temporal_encoding == "relative":
            history_t_emb = self.get_temporal_embedding(-lags).view(1, T, D).expand(B, -1, -1)
            current_t_emb = self.get_temporal_embedding(
                torch.zeros(B, device=memory.device, dtype=torch.float32)
            )
        else:
            history_t_emb = memory.new_zeros(B, T, D, dtype=torch.float32)
            current_t_emb = memory.new_zeros(B, D, dtype=torch.float32)

        history_t_emb = history_t_emb.to(dtype=memory.dtype)
        current_t_emb = current_t_emb.to(dtype=visual.dtype)
        
        # 处理每个层级
        new_memories = []
        for level in range(self.num_levels):
            level_memory = memory[:, level, :, :, :, :]   # [B, T, V, 64, D]
            level_visual = visual[:, level, :, :, :]       # [B, V, 64, D]
            
            if level not in self.injection_levels:
                new_memories.append(level_visual)
                continue

            level_memory_t = level_memory + history_t_emb.view(B, T, 1, 1, D)
            level_visual_t = level_visual + current_t_emb.view(B, 1, 1, D)
            
            # 聚合历史: [B, 5, V, 64, D] -> [B, V, 5*64, D]
            memory_per_view = level_memory_t.permute(0, 2, 1, 3, 4).reshape(B, V, T * S, D)

            valid_weights = valid_steps.to(level_memory_t.dtype).view(B, T, 1, 1, 1)
            mean_history = (level_memory_t * valid_weights).sum(dim=1)
            mean_history = mean_history / valid_weights.sum(dim=1).clamp_min(1.0)

            if self.use_slot_attention:
                if self.query_mode == "latest_history":
                    query = level_memory_t[:, -1]
                elif self.query_mode == "current_visual":
                    query = level_visual_t
                else:
                    query = mean_history
                token_mask = valid_steps[:, :, None].expand(B, T, S).reshape(B, T * S)
                history_agg = self.write_attns[level](query, memory_per_view, token_mask)
            else:
                history_agg = mean_history
            
            # 门控融合
            if self.fusion == "gated":
                gate_input = torch.cat([history_agg, level_visual_t], dim=-1)
                gate = self.write_gates[level](gate_input)
                level_new_memory = gate * level_visual_t + (1 - gate) * history_agg
            else:
                level_new_memory = level_visual_t + history_agg
            level_new_memory = self.norms[level](level_new_memory)
            
            new_memories.append(level_new_memory)
        
        # 堆叠: [B, 3, V, 64, D]
        new_memory = torch.stack(new_memories, dim=1)
        
        return new_memory


# ============== 使用示例 ==============

def demo():
    B, D = 4, 2560  # batch=4
    num_slots = 64
    T = 10
    device = torch.device('cpu')
    memory_bank = ShortTermMemoryBank(dim=D, num_slots=num_slots,num_timesteps=T,num_views=3).to(device)
    
    # 输入
    memory = torch.randn(B, 3, T, 3, num_slots, D).to(device)
    visual = torch.randn(B, 3, 3, num_slots, D).to(device)
    
    # 情况1: 所有样本相同 timestep (int)
    new_memory = memory_bank(memory, visual, timestep=10)
    print(f"Same timestep (int): {new_memory.shape}")
    
    # 情况2: batch 中不同 timestep [B]
    timesteps = torch.tensor([5, 10, 15, 20], dtype=torch.float32).to(device)
    new_memory = memory_bank(memory, visual, timestep=timesteps)
    print(f"Different timesteps [B]: {new_memory.shape}")
    
    # 情况3: scalar tensor
    new_memory = memory_bank(memory, visual, timestep=torch.tensor(7))
    print(f"Scalar tensor: {new_memory.shape}")


if __name__ == "__main__":
    demo()
