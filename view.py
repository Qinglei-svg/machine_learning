import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import logging as hf_logging
from transformers import AutoTokenizer

from train import CharGPT, get_config_by_mode


logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


def get_checkpoint_path(run_dir: Path, weight_type: str) -> Path:
    file_map = {
        "last": "last_model.pth",
        "best_test": "best_test_model.pth",
        "best_train": "best_train_model.pth",
    }
    ckpt_path = run_dir / file_map[weight_type]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到权重文件: {ckpt_path.as_posix()}")
    return ckpt_path


def list_available_modes(root_dir: Path) -> List[int]:
    modes = set()
    if not root_dir.exists():
        return []
    for p in root_dir.iterdir():
        if not p.is_dir():
            continue
        parts = p.name.split("_")
        if len(parts) == 3 and parts[0] == "train":
            try:
                modes.add(int(parts[1]))
            except ValueError:
                continue
    return sorted(modes)


def list_train_ids_by_mode(root_dir: Path, mode_size: int) -> List[int]:
    ids = []
    if not root_dir.exists():
        return []
    prefix = f"train_{mode_size}_"
    for p in root_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            try:
                ids.append(int(p.name.split("_")[-1]))
            except ValueError:
                continue
    return sorted(ids)


@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(
    mode_size: int,
    train_id: int,
    weight_type: str,
    device: str,
) -> Tuple[CharGPT, AutoTokenizer, Dict]:
    config = get_config_by_mode(mode_size)
    run_dir = Path("my_model") / f"train_{mode_size}_{train_id}"
    if not run_dir.exists():
        raise FileNotFoundError(f"未找到训练目录: {run_dir.as_posix()}")

    ckpt_path = get_checkpoint_path(run_dir, weight_type)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    tokenizer.add_tokens("<|e|>")
    tokenizer.end_ind = tokenizer.encode("<|e|>")[0]

    model = CharGPT(len(tokenizer), config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, {
        "config": config,
        "run_dir": run_dir.as_posix(),
        "ckpt_path": ckpt_path.as_posix(),
        "epoch": ckpt.get("epoch", "unknown"),
    }


@torch.no_grad()
def predict_next_tokens(
    model: CharGPT,
    tokenizer: AutoTokenizer,
    text: str,
    sequence_len: int,
    device: str,
    top_k: int,
    temperature: float,
) -> List[Tuple[str, int, float]]:
    if not text.strip():
        text = " "
    ids = tokenizer.encode(text)
    if len(ids) == 0:
        ids = [tokenizer.end_ind]

    context = torch.tensor(ids, device=device).unsqueeze(0)
    logits = model(context[:, -sequence_len:])
    next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
    probs = F.softmax(next_token_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

    result = []
    for token_id, prob in zip(top_ids[0].tolist(), top_probs[0].tolist()):
        token_str = tokenizer.decode([token_id])
        result.append((token_str, token_id, float(prob)))
    return result


@torch.no_grad()
def sample_next_token(
    model: CharGPT,
    tokenizer: AutoTokenizer,
    text: str,
    sequence_len: int,
    device: str,
    temperature: float,
) -> Tuple[str, int, float]:
    if not text.strip():
        text = " "
    ids = tokenizer.encode(text)
    if len(ids) == 0:
        ids = [tokenizer.end_ind]
    context = torch.tensor(ids, device=device).unsqueeze(0)
    logits = model(context[:, -sequence_len:])
    next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
    probs = F.softmax(next_token_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    next_id = sampled.item()
    next_prob = probs[0, next_id].item()
    return tokenizer.decode([next_id]), next_id, float(next_prob)


def main():
    parser = argparse.ArgumentParser(description="启动 token 预测可视化页面")
    parser.add_argument("--mode_size", type=int, default=1, help="参数组编号")
    parser.add_argument("--train_id", type=int, default=1, help="训练编号")
    parser.add_argument(
        "--weight_type",
        type=str,
        default="best_test",
        choices=["last", "best_test", "best_train"],
        help="加载权重类型",
    )
    parser.add_argument("--port", type=int, default=8501, help="页面端口")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="Token Predictor", layout="wide")
    st.markdown(
        """
        <style>
            .main-title {font-size: 28px; font-weight: 700; margin-bottom: 8px;}
            .sub-info {font-size: 13px; color: #6b7280;}
            .token-card {padding: 8px 10px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = Path("my_model")

    with st.sidebar:
        st.header("模型加载")
        available_modes = list_available_modes(root_dir)
        if not available_modes:
            st.error("未找到可用训练目录，请先运行 train.py")
            st.stop()

        default_mode = args.mode_size if args.mode_size in available_modes else available_modes[0]
        selected_mode = st.selectbox(
            "参数组 mode_size",
            options=available_modes,
            index=available_modes.index(default_mode),
        )

        train_ids = list_train_ids_by_mode(root_dir, selected_mode)
        if not train_ids:
            st.error(f"参数组 {selected_mode} 下没有训练记录。")
            st.stop()

        default_train_id = args.train_id if args.train_id in train_ids else train_ids[-1]
        selected_train_id = st.selectbox(
            "训练编号 train_id",
            options=train_ids,
            index=train_ids.index(default_train_id),
            help="对应目录: my_model/train_modeSize_trainId",
        )

        weight_options = ["last", "best_test", "best_train"]
        default_weight = args.weight_type if args.weight_type in weight_options else "best_test"
        selected_weight = st.selectbox(
            "权重文件",
            options=weight_options,
            index=weight_options.index(default_weight),
        )

    st.markdown('<div class="main-title">下一 Token 预测演示</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-info">device: {device} | mode_size={selected_mode} | train_id={selected_train_id} | weight={selected_weight}</div>',
        unsafe_allow_html=True,
    )

    try:
        model, tokenizer, meta = load_model_and_tokenizer(
            mode_size=selected_mode,
            train_id=selected_train_id,
            weight_type=selected_weight,
            device=device,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.caption(f"checkpoint: `{meta['ckpt_path']}` | epoch: `{meta['epoch']}`")

    if "current_text" not in st.session_state:
        st.session_state.current_text = "初秋的夜晚，祥子抬起头，"

    col_left, col_right = st.columns([1.35, 1.0], gap="large")
    with col_left:
        st.subheader("当前文本")
        st.session_state.current_text = st.text_area(
            "你可以手动编辑文本",
            value=st.session_state.current_text,
            height=380,
            label_visibility="collapsed",
        )
        if st.button("清空文本", use_container_width=True):
            st.session_state.current_text = ""
            st.rerun()

    with col_right:
        st.subheader("下一 Token 候选")
        top_k = st.slider("展示 Top-K", min_value=5, max_value=50, value=20, step=1)
        temperature = st.slider("Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
        auto_speed = st.slider("自动模拟速度（token/秒）", min_value=1, max_value=20, value=4, step=1)

        if "auto_run" not in st.session_state:
            st.session_state.auto_run = False
        if "last_auto_token_id" not in st.session_state:
            st.session_state.last_auto_token_id = None
        if "last_selected_token_info" not in st.session_state:
            st.session_state.last_selected_token_info = None

        try:
            candidates = predict_next_tokens(
                model=model,
                tokenizer=tokenizer,
                text=st.session_state.current_text,
                sequence_len=meta["config"].sequence_len,
                device=device,
                top_k=top_k,
                temperature=temperature,
            )
        except Exception as e:
            st.error(f"预测失败: {e}")
            st.stop()

        if st.session_state.last_selected_token_info is not None:
            info = st.session_state.last_selected_token_info
            st.markdown(
                (
                    "<div style='padding:10px 12px;border-radius:10px;"
                    "background:#fff1f2;border:2px solid #e11d48;"
                    "color:#9f1239;font-weight:700;margin:6px 0 12px 0;'>"
                    f"最近选中 Token: {info['token_show']} | id={info['token_id']} | "
                    f"p={info['prob']:.4f} | 来源: {info['source']}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

        for i, (token_str, token_id, prob) in enumerate(candidates, start=1):
            show_token = token_str.replace("\n", "\\n")
            is_auto_selected = token_id == st.session_state.last_auto_token_id
            marker = "🔥 " if is_auto_selected else ""
            if is_auto_selected:
                st.markdown(
                    (
                        "<div style='padding:8px 10px;border-radius:10px;"
                        "background:#fff1f2;border:2px solid #fb7185;"
                        "color:#9f1239;font-weight:700;margin:4px 0 8px 0;'>"
                        f"自动选中: {show_token} (token_id={token_id}, p={prob:.4f})"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                button_text = f"{marker}{i:02d}. 【自动选中】{show_token}  (p={prob:.4f})"
            else:
                button_text = f"{marker}{i:02d}. {show_token}  (p={prob:.4f})"
            if st.button(button_text, key=f"token_{i}", use_container_width=True):
                st.session_state.current_text += token_str
                st.rerun()
            if is_auto_selected:
                st.caption(f"token_id={token_id}  |  自动模式最近一次选中")
            else:
                st.caption(f"token_id={token_id}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("采样一个 Token", use_container_width=True):
                sampled, sampled_id, sampled_prob = sample_next_token(
                    model=model,
                    tokenizer=tokenizer,
                    text=st.session_state.current_text,
                    sequence_len=meta["config"].sequence_len,
                    device=device,
                    temperature=temperature,
                )
                st.session_state.current_text += sampled
                st.session_state.last_auto_token_id = sampled_id
                st.session_state.last_selected_token_info = {
                    "token_show": sampled.replace("\n", "\\n"),
                    "token_id": sampled_id,
                    "prob": sampled_prob,
                    "source": "手动采样",
                }
                st.rerun()
        with c2:
            auto_btn = "停止自动模拟" if st.session_state.auto_run else "开始自动模拟"
            if st.button(auto_btn, use_container_width=True):
                st.session_state.auto_run = not st.session_state.auto_run
                st.rerun()

        if st.session_state.auto_run:
            sampled, sampled_id, sampled_prob = sample_next_token(
                model=model,
                tokenizer=tokenizer,
                text=st.session_state.current_text,
                sequence_len=meta["config"].sequence_len,
                device=device,
                temperature=temperature,
            )
            st.session_state.current_text += sampled
            st.session_state.last_auto_token_id = sampled_id
            st.session_state.last_selected_token_info = {
                "token_show": sampled.replace("\n", "\\n"),
                "token_id": sampled_id,
                "prob": sampled_prob,
                "source": "自动模拟",
            }
            st.caption(f"自动模拟中... 当前速度: {auto_speed} token/秒")
            time.sleep(max(0.05, 1.0 / auto_speed))
            st.rerun()

    st.markdown("---")
    st.caption(
        "启动方式：`streamlit run view.py --logger.level error`（也可带默认值：`-- --mode_size 1 --train_id 1 --weight_type best_test`）"
    )


if __name__ == "__main__":
    main()
