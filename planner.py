"""Planner: action-sequence generator on top of frozen WM.

Given context + goal images, outputs N candidate action sequences via learned
queries (Perceiver-style self-attention). Best candidate is selected by
rollout cost. Diversity loss pushes non-best queries away from the best one
(DETR-style regularisation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class PlannerDecoder(nn.Module):
    """Perceiver-style decoder: N learned queries, cross-attend to [goal, ctx].

    Each query produces one action series of length ``horizon``.

    Architecture per layer:
        1. Self-attention among queries (diverse planning)
        2. Cross-attention: queries ← [goal_emb, ctx_emb]
        3. FFN

    Args:
        embed_dim: WM embedding dimension (192).
        num_queries: Number of action-plan candidates (N).
        num_layers: Transformer decoder layers.
        num_heads: Attention heads.
        mlp_dim: FFN hidden dimension.
        horizon: Length of output action series (T).
        action_dim: Raw action dimension per step.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_queries: int = 8,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        horizon: int = 5,
        action_dim: int = 2,
        action_substeps: int = 1,
        action_range: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_substeps = action_substeps
        self.embed_dim = embed_dim
        self.action_range = action_range

        # Learnable query tokens
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

        # Input projection for goal + ctx (already embed_dim, optional refine)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: query embedding → action sequence
        # Output horizon * action_substeps * action_dim individual scalar actions
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, horizon * action_substeps * action_dim),
        )
        # Small init: prevent sigmoid saturation at training start
        nn.init.normal_(self.action_head[2].weight, std=0.01)
        nn.init.zeros_(self.action_head[2].bias)

        # Confidence head: each query predicts how likely it is the right plan
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim // 4),
            nn.GELU(),
            nn.Linear(mlp_dim // 4, 1),
        )

    def forward(self, ctx_emb: torch.Tensor, goal_emb: torch.Tensor):
        """Produce N action sequences (Perceiver-style).

        Each query outputs ``horizon`` coarse steps. Each coarse step contains
        ``action_substeps`` individual raw actions (no repeat_interleave needed).

        Args:
            ctx_emb:  (B, HS, D)  context embedding (full history)
            goal_emb: (B, 1, D)   goal embedding

        Returns:
            actions: (B, N, horizon, action_substeps * action_dim)
        """
        B = ctx_emb.size(0)

        # Memory: [goal, ctx] — queries cross-attend to this
        memory = torch.cat([
            self.input_proj(goal_emb),
            self.input_proj(ctx_emb),
        ], dim=1)  # (B, 1 + HS, D)

        # Queries: N learnable tokens
        queries = self.query_embed.expand(B, -1, -1)  # (B, N, D)

        # Decoder: self-attn among queries, then cross-attn to [goal, ctx]
        out = self.decoder(tgt=queries, memory=memory)  # (B, N, D)

        # Predict action series per query — individual sub-actions
        actions = self.action_head(out)  # (B, N, horizon * action_substeps * action_dim)
        actions = actions.reshape(
            B, self.num_queries, self.horizon,
            self.action_substeps, self.action_dim,
        )
        actions = (2 * torch.sigmoid(actions) - 1) * self.action_range
        # Flatten substeps → (B, N, horizon, action_substeps * action_dim)
        actions = actions.reshape(
            B, self.num_queries, self.horizon,
            self.action_substeps * self.action_dim,
        )

        # Confidence per query (logits, BCEWithLogits applied in loss)
        conf = self.conf_head(out)  # (B, N, 1)

        return actions, conf


class PlannerLoss(nn.Module):
    """Best-of-N + diversity loss.

    - ``cost``: MSE between rollout embedding and goal embedding, per query.
    - ``loss``: cost of the *best* query (min cost).
    - ``diversity``: pushes non-best action sequences away from the best one.
    """

    def __init__(self, diversity_weight: float = 0.1, conf_weight: float = 0.5):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.conf_weight = conf_weight

    def forward(
        self,
        action_seqs: torch.Tensor,  # (B, N, T, action_substeps * action_dim)
        pred_embs: torch.Tensor,    # (B, N, T, D)  — per-step rollout embeddings
        goal_emb: torch.Tensor,     # (B, 1, D)     — final goal embedding
        conf: torch.Tensor = None,  # (B, N, 1)     — confidence per query
    ):
        """
        Returns:
            loss: scalar
            info: dict with cost, best_idx, diversity, conf terms
        """
        B, N, T, D = pred_embs.shape
        goal = goal_emb.reshape(B, 1, 1, D).expand(-1, N, T, -1)  # (B, N, T, D)

        # Per-query cost: mean MSE over all T rollout steps vs final goal
        costs = (pred_embs - goal).pow(2).mean(dim=-1).mean(dim=-1)  # (B, N)

        # Best query per sample
        best_cost, best_idx = costs.min(dim=-1)  # (B,), (B,)

        # ── Ortho loss: pairwise cos² penalty, all N×(N-1) query pairs ──
        flat_actions = action_seqs.reshape(B, N, -1)       # (B, N, T*A)
        flat_actions = F.normalize(flat_actions, dim=-1)   # unit vectors
        cos_mat = flat_actions @ flat_actions.transpose(-2, -1)  # (B, N, N)
        off_mask = ~torch.eye(N, dtype=torch.bool, device=action_seqs.device)
        div_loss = cos_mat[:, off_mask].pow(2).mean()      # push all off-diag → 0

        # ── Confidence loss (DETR-style: matched→1, unmatched→0) ──
        conf_loss = torch.tensor(0.0, device=action_seqs.device)
        if conf is not None:
            conf_target = torch.zeros_like(conf)  # (B, N, 1)
            conf_target[torch.arange(B), best_idx] = 1.0
            conf_loss = F.binary_cross_entropy_with_logits(
                conf.squeeze(-1), conf_target.squeeze(-1))

        loss = best_cost.mean() + self.diversity_weight * div_loss \
            + self.conf_weight * conf_loss

        info = {
            "loss": loss.detach(),
            "best_cost": best_cost.mean().detach(),
            "cost_mean": costs.mean().detach(),
            "cost_std": costs.std(dim=-1).mean().detach(),
            "diversity": div_loss.detach(),
            "conf_loss": conf_loss.detach(),
        }
        return loss, info


def planner_rollout(wm, actions, info_dict, history_size=3, hist_actions=None,
                    goal_emb=None, ctx_emb=None):
    """Run WM rollout using planner actions across N candidates.

    Args:
        wm: frozen JEPA model
        actions: (B, N, horizon, action_substeps * action_dim)  from planner (has grad)
        info_dict: dict with ``pixels`` (B, T_ctx, C, H, W), unused if ctx_emb given
        history_size: WM context length
        hist_actions: (B, HS, action_substeps * action_dim)  real past actions, or None
        goal_emb: (B, 1, D)  pre-computed goal embedding (required)
        ctx_emb: (B, T_ctx, D)  pre-computed context embedding (optional)

    Returns:
        pred_embs: (B, N, T, D)  per-step rollout embeddings for each candidate
        goal_emb:  (B, 1, D)
    """
    B, N, T, _ = actions.shape

    # Context encoding
    if ctx_emb is None:
        with torch.no_grad():
            out = wm.encode({k: v for k, v in info_dict.items()
                              if torch.is_tensor(v) and k != "goal"})
            ctx_emb = out["emb"]  # (B, T_ctx, D)

        if goal_emb is None:
            goal_emb = ctx_emb[:, -1:]  # fallback: last context frame

    # ── Rollout ──
    HS = history_size
    # actions are already (B, N, T, action_substeps * raw_dim) — no repeat needed
    act_dim = actions.shape[-1]

    ctx_emb_exp = ctx_emb.unsqueeze(1).expand(-1, N, -1, -1)
    emb = rearrange(ctx_emb_exp, "b n t d -> (b n) t d")[:, -HS:].clone()  # (B*N, HS, D)
    act_flat = rearrange(actions, "b n t d -> (b n) t d")  # (B*N, T, act_dim)

    # Build initial action history from real past actions, or zeros
    if hist_actions is not None:
        # hist_actions must be (B, HS, act_dim) — caller is responsible for this
        hist_expanded = hist_actions.unsqueeze(1).expand(-1, N, -1, -1)
        hist_act = rearrange(hist_expanded, "b n t d -> (b n) t d").clone()  # (B*N, HS, act_dim)
    else:
        hist_act = torch.zeros(B * N, HS, act_dim, device=actions.device)

    # Autoregressive rollout — collect per-step predictions
    step_preds = []  # one (B*N, D) per coarse step

    for t in range(T):
        cur_act = act_flat[:, t:t + 1].clone()  # clone: act_flat is a view into actions

        if t == 0:
            hist_act[:, -1:] = cur_act
        else:
            hist_act = torch.cat([hist_act[:, 1:], cur_act], dim=1)

        act_emb = wm.action_encoder(hist_act)
        pred = wm.predict(emb[:, -HS:], act_emb)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        step_preds.append(pred.squeeze(1))  # (B*N, D)

    # (T, B*N, D) → (B, N, T, D)
    pred_embs = rearrange(torch.stack(step_preds), "t (b n) d -> b n t d", b=B, n=N)

    return pred_embs, goal_emb
