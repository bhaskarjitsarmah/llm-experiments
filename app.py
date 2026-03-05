# @author: "Bhaskarjit"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformer_lens import HookedTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPT-2 Residual Stream Explorer",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading GPT-2 Small …")
def load_model():
    dev = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    m = HookedTransformer.from_pretrained("gpt2", device=dev)
    m.eval()
    return m, dev

model, device = load_model()
N_LAYERS = model.cfg.n_layers   # 12
N_HEADS  = model.cfg.n_heads    # 12
D_MODEL  = model.cfg.d_model    # 768
D_MLP    = model.cfg.d_mlp      # 3072
D_HEAD   = model.cfg.d_head     # 64
VOCAB    = model.cfg.d_vocab    # 50257
CTX_LEN  = model.cfg.n_ctx      # 1024


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    prompt = st.text_area(
        "Prompt",
        value="The Eiffel Tower is located in the city of",
        height=80,
    )
    target_word = st.text_input(
        "Target token (for logit lens)",
        value=" Paris",
        help="Must be a single token. Add a leading space for most words.",
    )
    st.divider()
    st.markdown("### GPT-2 Small")
    c1, c2 = st.columns(2)
    c1.metric("Layers",   N_LAYERS)
    c1.metric("d_model",  D_MODEL)
    c1.metric("Context",  CTX_LEN)
    c2.metric("Heads",    N_HEADS)
    c2.metric("d_mlp",    D_MLP)
    c2.metric("Vocab",    f"{VOCAB:,}")
    st.caption("d_head = d_model / n_heads = 64")


# ── Inference (cached in session state) ──────────────────────────────────────
@torch.no_grad()
def run_inference(p):
    toks     = model.to_tokens(p)
    tok_strs = model.to_str_tokens(p)
    logits, cache = model.run_with_cache(toks)
    return toks, tok_strs, logits, cache

if st.session_state.get("last_prompt") != prompt:
    with st.spinner("Running GPT-2 …"):
        _toks, _tok_strs, _logits, _cache = run_inference(prompt)
    st.session_state.update(
        last_prompt=prompt,
        tokens=_toks, token_strs=_tok_strs,
        logits=_logits, cache=_cache,
    )

tokens     = st.session_state["tokens"]
token_strs = st.session_state["token_strs"]
logits     = st.session_state["logits"]
cache      = st.session_state["cache"]
seq_len    = tokens.shape[1]

try:
    target_id    = model.to_single_token(target_word)
    target_valid = True
except Exception:
    target_valid = False
    st.sidebar.warning(f"'{target_word}' is not a single token — logit lens disabled.")


# ── Helpers ───────────────────────────────────────────────────────────────────
DARK_BG   = "#0f0f1a"
DARK_PLOT = "#16213e"
GRID_CLR  = "#2a2a4a"

def dark_layout(fig, height=400, **kwargs):
    fig.update_layout(
        height=height,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_PLOT,
        font=dict(color="white"),
        margin=dict(l=10, r=10, t=40, b=10),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID_CLR)
    fig.update_yaxes(gridcolor=GRID_CLR)
    return fig

@torch.no_grad()
def logit_lens_at(resid_1d):
    """Project a single residual vector (d_model) to vocab probabilities."""
    x = resid_1d.unsqueeze(0).unsqueeze(0)   # [1, 1, d_model]
    return torch.softmax(model.unembed(model.ln_final(x))[0, 0], dim=-1)

layer_labels = ["embed"] + [f"L{i}" for i in range(N_LAYERS)]


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔬 GPT-2 Small — Residual Stream Explorer")
tab1, tab2, tab3, tab4 = st.tabs([
    "🏗️ Architecture",
    "🌊 Residual Stream",
    "👁️ Attention Patterns",
    "🎯 Predictions",
])


# ═══════════════════════════════════════════
# TAB 1 — ARCHITECTURE DIAGRAM
# ═══════════════════════════════════════════
with tab1:
    st.subheader("GPT-2 Small — Full Pipeline")

    col_pipe, col_block = st.columns(2)

    # ── Left: full pipeline ──
    with col_pipe:
        shapes, annots = [], []

        def box(x0, y0, x1, y1, color, label, sub=""):
            shapes.append(dict(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                fillcolor=color, opacity=0.88,
                line=dict(color="white", width=1.5), layer="above",
            ))
            cy = (y0 + y1) / 2
            annots.append(dict(
                x=(x0+x1)/2, y=cy + (0.12 if sub else 0),
                text=f"<b>{label}</b>",
                showarrow=False, font=dict(color="white", size=12),
            ))
            if sub:
                annots.append(dict(
                    x=(x0+x1)/2, y=cy - 0.18,
                    text=f'<span style="font-size:10px;color:#ccc">{sub}</span>',
                    showarrow=False, font=dict(size=10),
                ))

        def arrow(x, ya, yb):
            annots.append(dict(
                x=x, y=yb, ax=x, ay=ya,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.2,
                arrowwidth=2, arrowcolor="#888",
            ))

        def plus(x, y):
            annots.append(dict(
                x=x, y=y, text="<b>⊕</b>",
                showarrow=False, font=dict(size=22, color="#68D391"),
            ))

        def stream_bar(y0, y1, label):
            shapes.append(dict(
                type="rect", x0=1.5, y0=y0, x1=8.5, y1=y1,
                fillcolor="#1e3a5f", opacity=0.85,
                line=dict(color="#4299E1", width=1.5),
            ))
            annots.append(dict(
                x=5, y=(y0+y1)/2,
                text=f'<span style="font-size:10px;color:#90CDF4">{label}</span>',
                showarrow=False,
            ))

        y = 0
        box(2, y, 8, y+0.75, "#4a4a6a", "Input Tokens",
            f"e.g.  {' | '.join(token_strs[:5])}")
        arrow(5, y+0.75, y+1.25)

        y = 1.25
        box(2, y, 8, y+0.75, "#2b6cb0", "Token Embedding  W_E",
            "[50,257 × 768]  — token ID → 768-dim vector")
        arrow(5, y+0.75, y+1.25)

        y = 2.5
        box(2, y, 8, y+0.75, "#2c5282", "Position Embedding  W_pos",
            "[1,024 × 768]  — position index → 768-dim vector")
        plus(5, 3.5)
        arrow(5, 3.72, 4.05)

        y = 4.05
        stream_bar(y, y+0.55, "Residual Stream  —  768-dim  —  the running sum")

        # Dashed block region
        y_blk = y + 0.55
        shapes.append(dict(
            type="rect", x0=1.5, y0=y_blk, x1=8.5, y1=y_blk+6.6,
            fillcolor="rgba(60,60,90,0.25)",
            line=dict(color="#718096", width=2, dash="dash"),
        ))
        annots.append(dict(
            x=5, y=y_blk+0.28,
            text="<b>× 12 Transformer Blocks</b>",
            showarrow=False, font=dict(color="#A0AEC0", size=12),
        ))

        # Inside one block
        yb = y_blk + 0.6
        box(2.5, yb, 7.5, yb+1.2, "#553c9a",
            "Multi-Head Attention",
            "12 heads × 64d  →  768d output")
        plus(5, yb+1.45)
        yb += 1.7
        stream_bar(yb, yb+0.45, "Residual Stream  (after attention)")
        yb += 0.65
        box(2.5, yb, 7.5, yb+1.2, "#c05621",
            "MLP",
            "768 → 3,072 → 768  (expand × 4, GeLU, compress)")
        plus(5, yb+1.45)
        yb += 1.7
        stream_bar(yb, yb+0.45, "Residual Stream  (after MLP)")

        arrow(5, y_blk+6.6, y_blk+7.1)
        y = y_blk + 7.2

        box(2, y, 8, y+0.75, "#276749", "Final Layer Norm", "")
        arrow(5, y+0.75, y+1.25)
        y += 1.25

        box(2, y, 8, y+0.75, "#2f855a", "Unembedding  W_U",
            "[768 × 50,257]  — vector → 50,257 logits")
        arrow(5, y+0.75, y+1.25)
        y += 1.25

        box(2, y, 8, y+0.75, "#285e61", "Softmax → Probabilities",
            'e.g.  P("Paris") = 0.41')

        fig_pipe = go.Figure()
        fig_pipe.update_layout(
            shapes=shapes, annotations=annots,
            xaxis=dict(range=[0, 10], visible=False),
            yaxis=dict(range=[y+1.2, -0.4], visible=False),
            height=750,
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            margin=dict(l=5, r=5, t=10, b=5),
        )
        st.plotly_chart(fig_pipe, use_container_width=True)

    # ── Right: zoom into the residual addition formula ──
    with col_block:
        st.markdown("### The Residual Addition Formula")
        st.markdown("""
The residual stream after all layers is a **pure sum**:

```
x  =  embed(tokens)            ← start
    + pos_embed(positions)
    + attn_out_layer_0(x)      ← layer 0 attention adds
    + mlp_out_layer_0(x)       ← layer 0 MLP adds
    + attn_out_layer_1(x)      ← layer 1 attention adds
    + mlp_out_layer_1(x)
    + ...
    + attn_out_layer_11(x)     ← layer 11
    + mlp_out_layer_11(x)
```

No layer **overwrites** — every component only **votes** by adding a vector.
The final answer emerges from 26 accumulated contributions.
        """)

        st.divider()
        st.markdown("### Dimension Breakdown")
        st.markdown(f"""
| Component | Shape | Role |
|-----------|-------|------|
| Token Embedding `W_E` | [50,257 × 768] | token ID → vector |
| Position Embedding | [1,024 × 768] | position → vector |
| Residual Stream | [seq_len × **768**] | the shared "whiteboard" |
| Q / K / V (per head) | [768 × 64] | project to head subspace |
| Attention output `W_O` | [64 × 768] × 12 | back to full space |
| MLP `W_in` | [768 × **3,072**] | expand × 4 |
| MLP `W_out` | [3,072 × 768] | compress back |
| Unembedding `W_U` | [768 × 50,257] | vector → logits |

> **Key identity**: `n_heads × d_head = 12 × 64 = 768 = d_model` ✓
        """)

        st.divider()
        st.markdown("### Parameter Count")
        total = sum(p.numel() for p in model.parameters())
        st.metric("Total parameters", f"{total/1e6:.1f} M")
        st.caption(
            "Most parameters live in the MLP layers "
            f"(2 × {N_LAYERS} × 768 × 3072 ≈ {2*N_LAYERS*768*3072/1e6:.0f}M) "
            "and the embedding / unembedding matrices."
        )


# ═══════════════════════════════════════════
# TAB 2 — RESIDUAL STREAM
# ═══════════════════════════════════════════
with tab2:
    st.subheader("Residual Stream Evolution")
    st.caption(f'Prompt: **"{prompt}"**')

    # ── Tokenization strip ──
    st.markdown("#### Tokenization")
    tok_cols = st.columns(len(token_strs))
    for i, (col, tok) in enumerate(zip(tok_cols, token_strs)):
        col.markdown(
            f"<div style='background:#1e3a5f;border-radius:6px;padding:5px 3px;"
            f"text-align:center;margin:2px'>"
            f"<div style='color:#90CDF4;font-size:10px'>pos {i}</div>"
            f"<div style='color:white;font-weight:bold;font-size:11px'>{repr(tok)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Residual stream L2-norm heatmap ──
    st.markdown("#### Residual Stream — L2 Norm Heatmap")
    st.caption(
        "Each cell shows the magnitude (L2 norm) of the 768-dim residual stream vector "
        "at a given layer and token position. Brighter = larger magnitude = more has been "
        "written to the stream at that point."
    )

    # build norm matrix [n_layers+1, seq]
    norm_rows = []
    norm_rows.append(
        np.linalg.norm(
            cache["resid_pre", 0][0].detach().cpu().numpy(), axis=-1
        )
    )
    for l in range(N_LAYERS):
        norm_rows.append(
            np.linalg.norm(
                cache["resid_post", l][0].detach().cpu().numpy(), axis=-1
            )
        )
    norm_mat = np.stack(norm_rows)   # [13, seq]

    fig_norm = go.Figure(go.Heatmap(
        z=norm_mat, x=token_strs, y=layer_labels,
        colorscale="Viridis",
        colorbar=dict(title="L2 Norm"),
        hovertemplate="Layer: %{y}<br>Token: %{x}<br>Norm: %{z:.2f}<extra></extra>",
    ))
    dark_layout(fig_norm, height=430,
                xaxis_title="Token", yaxis_title="Layer (residual stream checkpoint)")
    st.plotly_chart(fig_norm, use_container_width=True)

    # ── Layer contributions ──
    st.markdown("#### Layer Contributions — What does each component add?")
    st.caption(
        "L2 norm of the vector that each attention block and MLP adds to the residual "
        "stream at the **final token position** (which determines the next-token prediction)."
    )

    attn_norms, mlp_norms = [], []
    for l in range(N_LAYERS):
        attn_norms.append(float(np.linalg.norm(
            cache["attn_out", l][0, -1].detach().cpu().numpy()
        )))
        mlp_norms.append(float(np.linalg.norm(
            cache["mlp_out", l][0, -1].detach().cpu().numpy()
        )))

    fig_contrib = go.Figure()
    fig_contrib.add_trace(go.Bar(
        name="Attention", x=[f"L{i}" for i in range(N_LAYERS)], y=attn_norms,
        marker_color="#9F7AEA",
    ))
    fig_contrib.add_trace(go.Bar(
        name="MLP", x=[f"L{i}" for i in range(N_LAYERS)], y=mlp_norms,
        marker_color="#F6AD55",
    ))
    fig_contrib.update_layout(barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)"))
    dark_layout(fig_contrib, height=340,
                xaxis_title="Layer", yaxis_title="L2 norm of output vector")
    st.plotly_chart(fig_contrib, use_container_width=True)

    # ── Logit lens ──
    if target_valid:
        st.markdown(f"#### Logit Lens — Probability of `{target_word}` across layers")
        st.caption(
            "At each checkpoint we project the residual stream directly through the "
            "unembedding (skipping all remaining layers) and read off the probability "
            "of the target token. The orange labels show the top-1 predicted token at "
            "each layer."
        )

        probs, top_toks = [], []
        checkpoints = [cache["resid_pre", 0][0, -1]] + \
                      [cache["resid_post", l][0, -1] for l in range(N_LAYERS)]

        for resid in checkpoints:
            p_vec = logit_lens_at(resid)
            probs.append(p_vec[target_id].item())
            top_toks.append(model.to_single_str_token(p_vec.argmax().item()))

        fig_lens = go.Figure()
        fig_lens.add_trace(go.Scatter(
            x=layer_labels, y=probs,
            mode="lines+markers",
            marker=dict(size=9, color="#68D391", line=dict(color="white", width=1)),
            line=dict(color="#68D391", width=2.5),
            name=f'P("{target_word}")',
            hovertemplate="Layer %{x}<br>P = %{y:.4f}<extra></extra>",
        ))
        # Orange labels for top-1 token
        for lbl, p, tok in zip(layer_labels, probs, top_toks):
            fig_lens.add_annotation(
                x=lbl, y=p, yshift=18,
                text=f'<span style="color:#F6AD55;font-size:10px">{repr(tok)}</span>',
                showarrow=False, font=dict(size=10),
            )
        dark_layout(fig_lens, height=380,
                    xaxis_title="Layer",
                    yaxis_title=f'P("{target_word}")',
                    yaxis=dict(range=[0, max(probs) * 1.35], gridcolor=GRID_CLR))
        fig_lens.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_lens, use_container_width=True)

        # Prediction trail
        st.markdown("**Top-1 prediction trail** (embed → L11):")
        st.markdown("  →  ".join([f"`{repr(t)}`" for t in top_toks]))


# ═══════════════════════════════════════════
# TAB 3 — ATTENTION PATTERNS
# ═══════════════════════════════════════════
with tab3:
    st.subheader("Attention Pattern Explorer")
    st.caption(
        "Each cell (row i, col j) shows how much token i attends to token j. "
        "Brighter = stronger attention."
    )

    c_layer, c_head = st.columns(2)
    sel_layer = c_layer.slider("Layer", 0, N_LAYERS - 1, 0)
    sel_head  = c_head.slider("Head",  0, N_HEADS  - 1, 0)

    pat = cache["pattern", sel_layer][0, sel_head].detach().cpu().numpy()

    # ── Single head pattern ──
    fig_attn = go.Figure(go.Heatmap(
        z=pat, x=token_strs, y=token_strs,
        colorscale="Blues", colorbar=dict(title="Attention weight"),
        hovertemplate="From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>",
    ))
    dark_layout(fig_attn, height=420,
                title=f"Layer {sel_layer} · Head {sel_head}",
                xaxis_title="Key token (attends TO)",
                yaxis_title="Query token (attends FROM)")
    st.plotly_chart(fig_attn, use_container_width=True)

    # ── Head-type scores ──
    st.markdown("#### Head-type diagnostics for this head")
    prev_score  = float(np.diag(pat, k=-1).mean()) if seq_len > 1 else 0.0
    diag_score  = float(np.diag(pat).mean())
    first_score = float(pat[:, 0].mean())

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Prev-token score",   f"{prev_score:.3f}",
               help="High → attends mainly to the previous token")
    mc2.metric("Self-attention score", f"{diag_score:.3f}",
               help="High → attends to the current token")
    mc3.metric("First-token score",  f"{first_score:.3f}",
               help="High → always attends to position 0 (global sentinel)")

    # ── All 12 heads at this layer ──
    st.markdown(f"#### All 12 heads — Layer {sel_layer}")
    fig_all = make_subplots(rows=3, cols=4,
                            subplot_titles=[f"H{h}" for h in range(N_HEADS)],
                            horizontal_spacing=0.04, vertical_spacing=0.08)
    for h in range(N_HEADS):
        r, c = divmod(h, 4)
        p12 = cache["pattern", sel_layer][0, h].detach().cpu().numpy()
        fig_all.add_trace(
            go.Heatmap(z=p12, colorscale="Blues", showscale=False, hoverinfo="skip"),
            row=r + 1, col=c + 1,
        )
    fig_all.update_layout(
        height=520,
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_PLOT,
        font=dict(color="white", size=9),
        margin=dict(l=5, r=5, t=40, b=5),
    )
    st.plotly_chart(fig_all, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 4 — PREDICTIONS
# ═══════════════════════════════════════════
with tab4:
    st.subheader("Next Token Predictions")

    final_probs = torch.softmax(logits[0, -1].detach().cpu(), dim=-1)
    top20       = torch.topk(final_probs, 20)
    top_tokens  = [model.to_single_str_token(t.item()) for t in top20.indices]
    top_probs_v = top20.values.numpy()

    col_bar, col_list = st.columns([3, 2])

    # ── Bar chart ──
    with col_bar:
        st.markdown(f'#### Top-20 next-token predictions')
        st.caption(f'After prompt: *"{prompt}"*')
        fig_bar = go.Figure(go.Bar(
            x=top_probs_v,
            y=[repr(t) for t in top_tokens],
            orientation="h",
            marker=dict(
                color=top_probs_v, colorscale="Viridis",
                showscale=True, colorbar=dict(title="Prob"),
            ),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_bar.update_layout(yaxis=dict(autorange="reversed"))
        dark_layout(fig_bar, height=530, xaxis_title="Probability")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Ranked list ──
    with col_list:
        st.markdown("#### Top 10")
        max_p = float(top_probs_v[0])
        for i, (tok, p) in enumerate(zip(top_tokens[:10], top_probs_v[:10])):
            bar_pct = int(p / max_p * 100)
            st.markdown(
                f"<div style='margin:4px 0;padding:6px 10px;background:#1e2a3a;"
                f"border-radius:7px;display:flex;align-items:center;gap:10px'>"
                f"<span style='color:#718096;width:22px'>#{i+1}</span>"
                f"<span style='color:white;font-weight:bold;width:90px;"
                f"font-family:monospace'>{repr(tok)}</span>"
                f"<div style='flex:1;background:#2d3748;border-radius:4px;height:14px'>"
                f"<div style='width:{bar_pct}%;background:#68D391;"
                f"border-radius:4px;height:14px'></div></div>"
                f"<span style='color:#68D391;width:54px;text-align:right'>{p:.3f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Cross-layer comparison ──
    if target_valid:
        st.divider()
        st.markdown(f"#### Layer-by-layer: P(`{target_word}`) vs P(top-1 token)")
        st.caption(
            "Logit-lens view of how both the target token and the current top-1 "
            "token's probability evolve as the residual stream is built up."
        )

        p_target, p_top1, top1_names = [], [], []
        checkpoints = [cache["resid_pre", 0][0, -1]] + \
                      [cache["resid_post", l][0, -1] for l in range(N_LAYERS)]
        for resid in checkpoints:
            pv = logit_lens_at(resid)
            p_target.append(pv[target_id].item())
            best = pv.argmax().item()
            p_top1.append(pv[best].item())
            top1_names.append(model.to_single_str_token(best))

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=layer_labels, y=p_target,
            mode="lines+markers", name=f'P("{target_word}")',
            line=dict(color="#68D391", width=2.5),
            marker=dict(size=8),
        ))
        fig_cmp.add_trace(go.Scatter(
            x=layer_labels, y=p_top1,
            mode="lines+markers", name="P(top-1 token)",
            line=dict(color="#9F7AEA", width=2.5, dash="dash"),
            marker=dict(size=8),
            text=top1_names,
            hovertemplate="Layer %{x}<br>Top-1: %{text}<br>P = %{y:.4f}<extra></extra>",
        ))
        fig_cmp.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)"))
        dark_layout(fig_cmp, height=360,
                    xaxis_title="Layer",
                    yaxis_title="Probability",
                    yaxis=dict(gridcolor=GRID_CLR))
        st.plotly_chart(fig_cmp, use_container_width=True)
