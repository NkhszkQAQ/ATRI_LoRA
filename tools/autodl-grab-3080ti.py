#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autodl-grab-3080ti.py — AutoDL「北京B区 · RTX 3080 Ti」抢卡（算力市场下单版）

和 autodl-grab.py 的区别
------------------------
autodl-grab.py 抢的是**已有实例重新开机**（关机后卡被收回，轮询 power_on 抢回）。
这个脚本抢的是**算力市场里的 3080Ti 机器**——账号里还没有实例，抢到 = 直接下一单新实例。

为什么 3080Ti 要单独写
----------------------
3080Ti 不在容器实例 Pro API 的机型列表里（Pro 只有 5090/4090/3090/PRO6000 等），
所以只能走网页「算力市场」用的那套接口。这套接口实测（2026-11-04）：

  查库存  POST {MAIN}/api/v1/machine/search                     （要登录态）
  下单    POST {MAIN}/api/v1/order/instance/create/payg         （按量计费，下单即扣余额）
  查实例  GET  {MAIN}/api/v1/instance/exist?instance_uuid=...   （下单后轮询）
  余额    GET  {MAIN}/api/v1/wallet
  镜像    POST {MAIN}/api/v1/image/get

北京B区在算力市场里分三个 sign：bj-B1 / bj-B2 / bj-B3（控制台那边叫 beijingDC2，是同一区）。
可用机器判定（照抄网页表单的 disabled 逻辑）：
  on_line ∈ {3,4,5,6}  → 不可下单
  gpu_idle_num < 要的卡数 → 不够
  max_instance_num <= binding_instance_num → 满了

用法
----
  # 1. 先看一眼北京B区 3080Ti 现在什么情况（只读，不花钱）
  python3 tools/autodl-grab-3080ti.py stock

  # 2. 演练：有卡就打印「要下什么单」，但不真下（默认就是演练）
  python3 tools/autodl-grab-3080ti.py grab

  # 3. 真抢：盯着，一有能下单的机器就立刻下单（会花钱，按量 ¥1.03/时）
  python3 tools/autodl-grab-3080ti.py grab --yes

  # 常用变体
  python3 tools/autodl-grab-3080ti.py grab --yes --gpus 2 --name 3080ti-bjb
  python3 tools/autodl-grab-3080ti.py grab --yes --max-price 1.2 --interval 5
  python3 tools/autodl-grab-3080ti.py grab --yes --image base-image-b2343du52k

  # 后台挂着跑
  nohup python3 -u tools/autodl-grab-3080ti.py grab --yes >> logs/grab-3080ti.log 2>&1 &

前置条件
--------
- 浏览器登录态 token（和 autodl-grab.py 共用一份）：
    python3 tools/autodl-grab.py token
- 账号要实名认证、绑微信（AutoDL 下单要求），余额够（grab 会先查余额）

安全开关
--------
- 不加 --yes 一律只演练，不扣钱
- --max-price 默认 ¥1.5/时，超过这个价的机器不碰
- 下单失败（被别人抢走 / 余额不足 / 未实名）会打印错误码，抢走的继续蹲下一台
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

API_MAIN = "https://www.autodl.com"
CONFIG_DIR = os.path.expanduser("~/.config/autodl-grab")
TOKEN_FILE = os.path.join(CONFIG_DIR, "token")
APP_VERSION = "v6.32.0"

REGION_SIGNS = ["bj-B1", "bj-B2", "bj-B3"]   # 三个 sign 都属于「北京B区」
DEFAULT_GPU = "RTX 3080 Ti"
# PyTorch 2.5.1 / Python3.12 / CUDA12.4：3080Ti(Ampere) 和后续训练都能用
DEFAULT_IMAGE = "base-image-b2343du52k"
DEFAULT_MAX_PRICE = 1.5      # 元/时
DEFAULT_GPUS = 1
DEFAULT_INTERVAL = 10        # 轮询间隔秒


# ────────────────────────────────────────────── 基础工具 ──

def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


class ApiError(Exception):
    pass


def _read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def get_token() -> str:
    tok = os.environ.get("AUTODL_TOKEN", "").strip() or _read_file(TOKEN_FILE)
    if not tok:
        sys.exit(
            "缺少 token。先跑：python3 tools/autodl-grab.py token\n"
            "（浏览器登录 autodl.com → F12 → Network → 复制任意 api/v1 请求的 Authorization）"
        )
    return tok


def api(path: str, method: str = "POST", body=None, params=None,
        timeout: int = 25) -> dict:
    """发 JSON 请求；强制直连（国内站走代理反而失败）。"""
    url = API_MAIN + path
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "AppVersion": APP_VERSION,
        "User-Agent": "autodl-grab-3080ti/1.0",
    }
    tok = get_token()
    headers["Authorization"] = tok
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    elif method in ("POST", "PUT"):
        data = b"{}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise ApiError(f"HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise ApiError(f"网络错误：{e.reason}") from e
    except TimeoutError:
        raise ApiError("请求超时")
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ApiError(f"响应不是 JSON：{raw[:200]}") from e
    if d.get("code") not in ("Success",):
        raise ApiError(f"{d.get('code')} {d.get('msg')}")
    return d


def notify(title: str, message: str) -> None:
    if shutil.which("osascript"):
        try:
            script = (
                f"display notification {json.dumps(message, ensure_ascii=False)} "
                f"with title {json.dumps(title, ensure_ascii=False)} "
                'sound name "Glass"'
            )
            subprocess.run(["osascript", "-e", script], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print("\a", end="", flush=True)


# ────────────────────────────────────────────── 接口 ──

def get_balance_yuan() -> float:
    d = api("/api/v1/wallet", method="GET")
    assets = (d.get("data") or {}).get("assets") or 0
    return assets / 1000.0


def search_machines(gpu: str, signs: list[str]) -> list[dict]:
    """市场里搜机器。page_size 给大点，北京B区 3080Ti 十几台一次够。"""
    body = {
        "charge_type": "payg",
        "region_sign": "",
        "gpu_type_name": [gpu],
        "machine_tag_name": [],
        "gpu_idle_num": 0,
        "mount_net_disk": False,
        "instance_disk_size_order": "",
        "date_range": "",
        "date_from": "",
        "date_to": "",
        "page_index": 1,
        "page_size": 200,
        "pay_price_order": "",
        "gpu_idle_type": "",
        "default_order": True,
        "region_sign_list": signs,
    }
    d = api("/api/v1/machine/search", body=body)
    return (d.get("data") or {}).get("list") or []


def can_order(m: dict, need_gpus: int) -> bool:
    """照抄网页表单的 disabled 逻辑。"""
    if m.get("on_line") in (3, 4, 5, 6):
        return False
    if (m.get("gpu_idle_num") or 0) < need_gpus:
        return False
    if (m.get("max_instance_num") or 0) <= (m.get("binding_instance_num") or 0):
        return False
    return True


def price_of(m: dict) -> float:
    return (m.get("payg_price") or 0) / 1000.0


def order_payg(machine_id: str, gpus: int, image_uuid: str, name: str) -> dict:
    """按量计费下单。成功返回 data（里面有订单/实例 uuid）。"""
    body = {
        "instance_info": {
            "machine_id": machine_id,
            "charge_type": "payg",
            "req_gpu_amount": gpus,
            "base_image_uuid": image_uuid,
            "private_image_uuid": "",
            "reproduction_uuid": "",
            "cg_application_uuid": "",
            "cg_application_info": {
                "image_name": "", "current_version_id": 0,
                "current_version": "", "image_id": 0,
            },
            "instance_name": name,
            "expand_data_disk": 0,
            "reproduction_id": 0,
        },
        "price_info": {
            "coupon_id_list": [],
            "machine_id": machine_id,
            "charge_type": "payg",
            "duration": 1,
            "num": gpus,
            "expand_data_disk": 0,
        },
    }
    return api("/api/v1/order/instance/create/payg", body=body)


def instance_exists(instance_uuid: str) -> bool:
    try:
        d = api("/api/v1/instance/exist", method="GET",
                params={"instance_uuid": instance_uuid})
    except ApiError:
        return False
    return bool(d.get("data"))


def list_instances() -> list[dict]:
    body = {
        "page_index": 1, "page_size": 100, "status": [], "charge_type": [],
        "sub_name": "", "unbind_sub_user": False,
    }
    d = api("/api/v1/instance", body=body)
    return (d.get("data") or {}).get("list") or []


# ────────────────────────────────────────────── 子命令 ──

def cmd_stock(args) -> int:
    rows = search_machines(args.gpu, args.region)
    if not rows:
        print(f"北京B区没搜到 {args.gpu}。")
        return 0
    print(f"AutoDL 算力市场 · {' + '.join(args.region)} · {args.gpu} · {now()}")
    print(f"{'机器':<10}{'区':<8}{'在线':>4}{'空闲/总卡':>12}{'价格':>10}   状态")
    print("-" * 62)
    orderable = 0
    for m in sorted(rows, key=lambda x: (-(x.get("gpu_idle_num") or 0),
                                         x.get("payg_price") or 0)):
        idle = m.get("gpu_idle_num") or 0
        total = m.get("gpu_number") or 0
        ok = can_order(m, args.gpus)
        if ok:
            orderable += 1
        state = "🟢 可下单" if ok else ("⚪ 无空闲" if idle == 0 else "🔒 不可选")
        print(f"{str(m.get('machine_alias')):<10}"
              f"{str(m.get('region_sign')):<8}"
              f"{str(m.get('on_line')):>4}"
              f"{f'{idle}/{total}':>12}"
              f"{f'¥{price_of(m):.2f}/时':>10}   {state}")
    print("-" * 62)
    print(f"共 {len(rows)} 台，其中现在能下单的 {orderable} 台"
          f"（要 {args.gpus} 张卡，价格 ≤ ¥{args.max_price:.2f}/时）")
    if orderable:
        print("下单：python3 tools/autodl-grab-3080ti.py grab --yes")
    return 0


def cmd_grab(args) -> int:
    name = args.name or f"3080ti-bjb-{datetime.now().strftime('%m%d-%H%M')}"
    log(f"目标：{args.gpu} · 北京B区({'/'.join(args.region)}) · {args.gpus} 张卡 · "
        f"价格上限 ¥{args.max_price:.2f}/时 · 轮询 {args.interval}s")
    try:
        bal = get_balance_yuan()
    except ApiError as e:
        sys.exit(f"查余额失败：{e}")
    log(f"账户余额 ¥{bal:.2f}")
    if bal < 10:
        log(f"⚠️ 余额偏低（¥{bal:.2f}），下单可能返回 TORecharge")

    if not args.yes:
        log("演练模式：只找能下单的机器、打印订单参数，不会花钱。")
        log("确认要真下单，加 --yes。")

    tries = 0
    while True:
        try:
            rows = search_machines(args.gpu, args.region)
        except ApiError as e:
            log(f"查库存失败：{e}")
            time.sleep(args.interval)
            continue

        cands = [m for m in rows
                 if can_order(m, args.gpus) and price_of(m) <= args.max_price]
        cands.sort(key=lambda m: (price_of(m), -(m.get("gpu_idle_num") or 0)))

        if not cands:
            log(f"暂无可以下单的机器（{len(rows)} 台里 0 台满足条件）")
            if args.max_tries and tries >= args.max_tries:
                log("按 --max-tries 放弃。")
                return 1
            tries += 1
            time.sleep(args.interval * random.uniform(0.9, 1.1))
            continue

        m = cands[0]
        log(f"🟢 发现可下单：{m.get('machine_alias')}（{m.get('region_name')}/"
            f"{m.get('region_sign')}）空闲 {m.get('gpu_idle_num')} 张，"
            f"¥{price_of(m):.2f}/时")
        log(f"   准备下单：machine_id={m.get('machine_id')} "
            f"gpus={args.gpus} image={args.image} name={name}")

        if not args.yes:
            log("   （演练模式，未下单。加 --yes 才会真扣钱）")
            return 0

        try:
            r = order_payg(m["machine_id"], args.gpus, args.image, name)
        except ApiError as e:
            msg = str(e)
            if msg.startswith(("MachineGpuNumUpdateFailed",)):
                log(f"   ✗ 机器被抢走了（{msg}），继续蹲。")
                tries += 1
                time.sleep(args.interval)
                continue
            if msg.startswith("TORecharge"):
                sys.exit(f"✗ 余额不足：{msg}")
            if msg.startswith("TORealName"):
                sys.exit(f"✗ 需要实名认证/绑微信：{msg}")
            if msg.startswith("OrderUnpaidMoreThanThree"):
                sys.exit(f"✗ 有 3 个未支付订单，先去付掉：{msg}")
            log(f"   ✗ 下单失败：{msg}")
            tries += 1
            time.sleep(args.interval)
            continue

        data = r.get("data")
        uuid = data.get("uuid") if isinstance(data, dict) else data
        log(f"✅ 下单成功，订单/实例：{uuid}")

        # 轮询等实例真的出现
        found = None
        for _ in range(40):
            time.sleep(3)
            try:
                ins = [x for x in list_instances()
                       if str(x.get("uuid")) == str(uuid) or
                       (name and x.get("name") == name)]
            except ApiError:
                ins = []
            if ins:
                found = ins[0]
                log(f"实例已出现：{found.get('status')}（{found.get('ssh_command', '')}）")
                break
            if uuid and instance_exists(str(uuid)):
                log("实例已创建（exist 接口确认），等它进入 running ...")
                continue
        if not found:
            log("⚠️ 实例还没在列表里出现，稍后自己看控制台。")

        notify("AutoDL 3080Ti 抢到了", f"{name} 已下单：{uuid}")
        print()
        print("接下来：")
        print(f"  ssh 命令：{(found or {}).get('ssh_command') or '去控制台看'}")
        print("  按量计费，跑完记得关机。")
        return 0


# ────────────────────────────────────────────── 入口 ──

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoDL 北京B区 RTX 3080 Ti 抢卡（算力市场下单）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""例子:
  python3 tools/autodl-grab-3080ti.py stock
  python3 tools/autodl-grab-3080ti.py grab
  python3 tools/autodl-grab-3080ti.py grab --yes --gpus 1 --name 3080ti-bjb
""")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--gpu", default=DEFAULT_GPU, help=f"GPU 型号，默认 {DEFAULT_GPU}")
        s.add_argument("--region", nargs="+", default=REGION_SIGNS,
                       help="区域 sign，默认 bj-B1 bj-B2 bj-B3（都是北京B区）")
        s.add_argument("--gpus", type=int, default=DEFAULT_GPUS, help="要几张卡，默认 1")
        s.add_argument("--max-price", type=float, default=DEFAULT_MAX_PRICE,
                       help=f"价格上限（元/时），默认 {DEFAULT_MAX_PRICE}")

    s = sub.add_parser("stock", help="看北京B区 3080Ti 库存（只读）")
    common(s)
    s.set_defaults(func=cmd_stock)

    s = sub.add_parser("grab", help="蹲库存，有可下单的机器就下（默认演练，不花钱）")
    common(s)
    s.add_argument("--yes", action="store_true", help="真下单（会扣钱），不加只演练")
    s.add_argument("--image", default=DEFAULT_IMAGE,
                   help=f"基础镜像 uuid，默认 {DEFAULT_IMAGE}（PyTorch2.5.1/cu124）")
    s.add_argument("--name", default=None, help="实例备注名，默认自动按时间生成")
    s.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="轮询间隔秒")
    s.add_argument("--max-tries", type=int, default=0, help="查多少次没货就放弃，0=一直蹲")
    s.set_defaults(func=cmd_grab)

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
