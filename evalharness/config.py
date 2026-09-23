# -*- coding: utf-8 -*-
"""Config Snapshot —— run_id 由「配置内容 + 决策逻辑文件内容」共同决定。

为什么这是全项目第一个要写的模块
--------------------------------
项目一的 run_summary_*.json 只有 7 个字段：没有 model_name、没有配置指纹。
两组跑批的差异只能靠目录名后缀来证明 —— 那不构成受控实验。

这条规则本身就是评测框架设计的一条原则：
  改了 prompt 或换了模型，run_id 必须变；否则两次实验会被错误地归为同一次。
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# 参与 hash 的全部字段。任何一项变了，run_id 就必须变。
# 分四类，因为"变的是哪一类"决定了你该怎么解释结果：
#   模型侧 变了 -> 结论只能说明"某模型更好"，不能归因到 prompt
#   代码侧 变了 -> 决策逻辑变了
#   场景侧 变了 -> 两组数据根本不可比
#   预算侧 变了 -> 结果可能是被熔断截断的，不是真实能力
SNAPSHOT_FIELDS: tuple[str, ...] = (
    # 模型侧
    "model_name",
    "model_api_url",
    "temperature",
    # 代码侧
    "prompt_version",
    "adapter_name",
    # ★ variant 必须进快照。早期设计文档的 SNAPSHOT_FIELDS 漏了它 —— 那是个真 bug：
    # 消融实验里"基线"和"消融变体"如果只有 variant 不同、别的字段全一样，
    # 两者的 run_id 会相同，于是两次实验会被错误地归为同一次 (见 docs/architecture.md)。
    "variant",
    # 场景侧
    "simulation_duration_days",
    "simulation_max_steps",
    # 预算侧
    "max_tokens_per_driver",
    "max_wall_time_minutes",
)


def _git(args: str, cwd: Path) -> str:
    """取 git 信息。任何失败都返回空串。

    为什么不抛异常：快照工具本身绝不能因为"这台机器没装 git"而让整个实验崩掉。
    """
    try:
        out = subprocess.run(
            ["git", *args.split()],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def hash_files(paths: Sequence[Path | str]) -> str:
    """决策逻辑文件的内容哈希。

    为什么不能只哈希配置：配置文件一样、但代码改了，两次结果同样不可比。
    这是"踩过一次坑之后加的"那一行 —— 面试里这句话比代码本身值钱。
    """
    if not paths:
        return "no-impl-files"
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        # 把文件名也喂进去：否则"删掉一个文件"这种改动不会改变哈希
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class ConfigSnapshot:
    """一次运行的配置快照。落盘后，任何一次实验都可被精确复述与比对。"""

    run_id: str
    fields: dict[str, Any]
    impl_hash: str
    git_sha: str
    git_dirty: bool
    python_version: str

    @classmethod
    def capture(
        cls,
        cfg: Mapping[str, Any],
        impl_paths: Sequence[Path | str] = (),
        repo_dir: Path | str = ".",
    ) -> "ConfigSnapshot":
        repo = Path(repo_dir)
        fields = {k: cfg.get(k) for k in SNAPSHOT_FIELDS}
        impl_hash = hash_files(impl_paths)

        payload = json.dumps(
            {"fields": fields, "impl": impl_hash},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        run_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

        return cls(
            run_id=run_id,
            fields=fields,
            impl_hash=impl_hash,
            git_sha=_git("rev-parse --short HEAD", repo),
            # 有未提交改动时必须标记 —— 否则 sha 是假的，指向一个不等于当前代码的版本
            git_dirty=bool(_git("status --porcelain", repo)),
            python_version=platform.python_version(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fields": self.fields,
            "impl_hash": self.impl_hash,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)


def diff(a: ConfigSnapshot, b: ConfigSnapshot) -> dict[str, tuple[Any, Any]]:
    """两组跑批的差异报告 —— 这一句就是面试答案：

        「我能精确告诉你两组之间变了哪几个字段，其余全部相同。」

    返回值只包含真正变化的字段；没有变化时返回空 dict，
    这意味着"两次运行配置完全相同，结果差异只能来自随机性"—— 这同样是重要结论。
    """
    changed: dict[str, tuple[Any, Any]] = {
        k: (a.fields.get(k), b.fields.get(k))
        for k in SNAPSHOT_FIELDS
        if a.fields.get(k) != b.fields.get(k)
    }
    if a.impl_hash != b.impl_hash:
        changed["__decision_logic__"] = (a.impl_hash, b.impl_hash)
    return changed
