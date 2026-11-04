#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autodl-grab.py — AutoDL 抢卡 / 开机小工具（单文件，零第三方依赖）

背景
----
按量计费的 AutoDL 实例一关机，GPU 就被平台收回；再开机时要等机房有空闲卡，
官方没有排队，纯先到先得——谁先调开机接口谁拿到。别人（或别人的脚本）在秒级
轮询，所以你手动点必然抢不到。这个脚本替你在后台按固定节奏重试开机，抢到为止。

真实 API（2026-11 实测）
----------------------
  库存查询（匿名，不用 token）:
    POST https://www.autodl.art/api/v1/adl_server/machine_pro/gpu_type
    body: {"data_center_list": ["westDC3", "beijingDC2"]}
  两套鉴权（按优先级自动探测）:
    1) 开发者 Token（推荐，长期有效）: 控制台 → 账号 → 设置 → 开发者Token
       Pro 实例: POST /api/v1/dev/instance/pro/{list,power_on,power_off}
       普通实例: POST /api/v1/dev/instance/{list,power_on,power_off}（文档未公开，通了才用）
    2) 浏览器 JWT（兜底，会过期）:
       POST https://www.autodl.com/api/v1/instance
       POST https://www.autodl.com/api/v1/instance/power_on  body: {"instance_uuid": "<uuid>"}

用法
----
  # 1. 查实时库存（匿名，随时可用）
  python3 autodl-grab.py stock
  python3 autodl-grab.py stock --gpu 5090 --region westDC3 beijingDC2

  # 2a. 保存开发者 Token（首选：控制台 → 账号 → 设置 → 开发者Token，长期有效）
  python3 autodl-grab.py api

  # 2b. 备用：保存浏览器里的 Authorization（会过期，仅在开发者 Token 用不了时用）
  python3 autodl-grab.py token

  # 3. 看自己的实例（拿 uuid）
  python3 autodl-grab.py list

  # 4. 轮询开机，抢到为止（假设已经手动关好机）
  python3 autodl-grab.py poweron <uuid>
  python3 autodl-grab.py poweron <uuid> --interval 15

  # 5. 一键抢卡：如果无卡模式在跑，先关机再死等有卡（最省事）
  python3 autodl-grab.py grab <uuid>

  # 6. 盯库存，有货响铃（可选顺带自动开机）
  python3 autodl-grab.py watch --gpu 5090
  python3 autodl-grab.py watch --gpu 5090 --poweron <uuid>
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
import urllib.request
from datetime import datetime

API_ART = "https://www.autodl.art"
API_MAIN = "https://www.autodl.com"
CONFIG_DIR = os.path.expanduser("~/.config/autodl-grab")
TOKEN_FILE = os.path.join(CONFIG_DIR, "token")          # 浏览器 JWT（兜底）
API_TOKEN_FILE = os.path.join(CONFIG_DIR, "api_token")  # 控制台开发者 Token（优先）
APP_VERSION = "v6.32.0"          # 网页 localStorage.app_version，服务端不严格校验
DEFAULT_REGIONS = ["westDC3", "beijingDC2"]   # 西北区 + 北京区
REGION_NAMES = {"westDC3": "西北区", "beijingDC2": "北京区"}
DEFAULT_MAIL_TO = [a.strip() for a in os.environ.get("AUTODL_GRAB_MAIL_TO", "").split(",") if a.strip()]  # 通知收件人，逗号分隔；不设就不发邮件
DEFAULT_INTERVAL = 5  # 轮询间隔秒（主人指定：5 秒）


# ────────────────────────────────────────────── 基础工具 ──

def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


class ApiError(Exception):
    pass


def http_json(method: str, url: str, body=None, token: str | None = None,
              timeout: int = 15) -> dict:
    """发 JSON 请求；强制直连（AutoDL 是国内站，走代理反而容易失败）。"""
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "AppVersion": APP_VERSION,
        "User-Agent": "autodl-grab/1.0",
    }
    if token:
        headers["Authorization"] = token
    data = None if body is None else json.dumps(body).encode("utf-8")
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
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ApiError(f"响应不是 JSON：{raw[:200]}") from e


def notify(title: str, message: str) -> None:
    """macOS 通知 + 提示音；没权限就退化为终端响铃。"""
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


def _as_applescript_string(text: str) -> str:
    """把 Python 字符串转成 AppleScript 双引号字符串（安全转义）。"""
    return json.dumps(text, ensure_ascii=False)


def send_mail(subject: str, body_lines: list[str],
              to: list[str] | None = None) -> bool:
    """用 Mail.app（AppleScript）发邮件。成功返回 True。"""
    recipients = to if to is not None else DEFAULT_MAIL_TO
    if not recipients:
        return False
    content = " & return & ".join(_as_applescript_string(x) for x in body_lines)
    to_lines = "\n        ".join(
        f"make new to recipient at end of to recipients with properties "
        f"{{address:{_as_applescript_string(a)}}}"
        for a in recipients
    )
    script = f"""tell application "Mail"
    set newMsg to make new outgoing message with properties {{subject:{_as_applescript_string(subject)}, content:({content}), visible:false}}
    tell newMsg
        {to_lines}
        send
    end tell
end tell
"""
    ok = False
    for attempt in range(2):
        try:
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                ok = True
                break
            err = (r.stderr or "").strip()
            if err:
                log(f"邮件发送失败（第 {attempt + 1} 次）：{err[:160]}")
        except Exception as e:
            log(f"邮件发送异常：{e}")
            break
    return ok


def _read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def require_auth() -> tuple[str, str]:
    """返回 (凭据, 模式)。优先级：api-pro → api → jwt。

    api-pro = 开发者 Token + 容器实例 Pro 接口（官方文档公开，长期不过期）
    api     = 开发者 Token + 普通实例 dev 接口（文档未公开，探测通了才用）
    jwt     = 浏览器登录态（会过期，兜底）
    """
    api_tok = os.environ.get("AUTODL_API_TOKEN", "").strip() or _read_file(API_TOKEN_FILE)
    if api_tok:
        try:
            r = http_json("POST", API_MAIN + "/api/v1/dev/instance/pro/list",
                          {"page_index": 1, "page_size": 1}, api_tok)
            if r.get("code") == "Success":
                return api_tok, "api-pro"
        except ApiError:
            pass
        try:
            r = http_json("POST", API_MAIN + "/api/v1/dev/instance/list",
                          {"page_index": 1, "page_size": 1}, api_tok)
            if r.get("code") == "Success":
                return api_tok, "api"
        except ApiError:
            pass
        log("⚠️ 开发者 Token 调不通实例接口（可能要实名认证，或实例不是 Pro 类型），回退浏览器 token")

    jwt = os.environ.get("AUTODL_TOKEN", "").strip() or _read_file(TOKEN_FILE)
    if not jwt:
        sys.exit(
            "缺少凭据。推荐先运行 `python3 autodl-grab.py api` 保存开发者 Token；"
            "备用 `python3 autodl-grab.py token` 保存浏览器 token。"
        )
    return jwt, "jwt"


# ────────────────────────────────────────────── 接口调用 ──

def query_stock(regions: list[str]) -> list[dict]:
    r = http_json("POST", API_ART + "/api/v1/adl_server/machine_pro/gpu_type",
                  {"data_center_list": regions})
    if r.get("code") != "Success":
        raise ApiError(f"{r.get('code')} {r.get('msg')}")
    return r.get("data") or []


def list_instances(token: str, mode: str = "jwt") -> list[dict]:
    if mode == "api-pro":
        url, body = API_MAIN + "/api/v1/dev/instance/pro/list", {"page_index": 1, "page_size": 100}
    elif mode == "api":
        url, body = API_MAIN + "/api/v1/dev/instance/list", {"page_index": 1, "page_size": 100}
    else:
        url = API_MAIN + "/api/v1/instance"
        body = {
            "page_index": 1,
            "page_size": 100,
            "status": [],
            "charge_type": [],
            "sub_name": "",
            "unbind_sub_user": False,
        }
    r = http_json("POST", url, body, token)
    if r.get("code") != "Success":
        raise ApiError(f"{r.get('code')} {r.get('msg')}")
    return (r.get("data") or {}).get("list") or []


def find_instance(token: str, uuid: str, mode: str = "jwt") -> dict | None:
    for ins in list_instances(token, mode):
        if ins.get("uuid") == uuid:
            return ins
    return None


def power_on(token: str, uuid: str, mode: str = "jwt") -> dict:
    if mode == "api-pro":
        return http_json("POST", API_MAIN + "/api/v1/dev/instance/pro/power_on",
                         {"instance_uuid": uuid, "payload": "gpu"}, token)
    if mode == "api":
        return http_json("POST", API_MAIN + "/api/v1/dev/instance/power_on",
                         {"instance_uuid": uuid}, token)
    return http_json("POST", API_MAIN + "/api/v1/instance/power_on",
                     {"instance_uuid": uuid}, token)


# ────────────────────────────────────────────── 子命令 ──

def cmd_stock(args) -> None:
    regions = args.region or DEFAULT_REGIONS
    rows = query_stock(regions)
    kws = [k.strip().lower() for k in (args.gpu or []) if k.strip()]
    if kws:
        rows = [g for g in rows if any(k in (g.get("user_define_name") or "").lower() for k in kws)]
    region_label = " + ".join(REGION_NAMES.get(r, r) for r in regions)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if not rows:
        print(f"没有匹配的 GPU 型号（筛选：{', '.join(kws) or '全部'}）")
        return

    print(f"AutoDL 实时库存 · {region_label} · {now()}")
    print(f"{'型号':<16}{'空闲':>6}{'总数':>8}   状态")
    print("-" * 48)
    idle_total = 0
    for g in sorted(rows, key=lambda x: -(x.get("gpu_idle_num") or 0)):
        name = g.get("user_define_name") or "?"
        idle = g.get("gpu_idle_num") or 0
        total = g.get("gpu_total_number") or 0
        idle_total += idle
        state = "🟢 有货" if idle > 0 else "⚪ 无"
        print(f"{name:<16}{idle:>6}{total:>8}   {state}")
    print("-" * 48)
    print(f"空闲合计：{idle_total} 张  |  匿名接口，无需登录")


def cmd_token(args) -> None:
    print("获取 Authorization（浏览器登录态，不是开发者 Token）：")
    print("  1) 浏览器打开并登录 https://www.autodl.com")
    print("  2) F12 → Network → 点任意一个 api/v1 请求")
    print("  3) Request Headers → 复制 Authorization 的值（eyJ 开头的一长串）")
    print()
    tok = (args.token_value or "").strip()
    if not tok:
        tok = input("粘贴 token: ").strip()
    if not tok:
        sys.exit("没输入 token，放弃。")

    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(tok)
    os.chmod(TOKEN_FILE, 0o600)
    log(f"已保存（600）→ {TOKEN_FILE}")

    # 顺手验证一下
    try:
        ins = list_instances(tok, "jwt")
        log(f"验证通过：账号下有 {len(ins)} 个实例")
    except ApiError as e:
        log(f"⚠️ 验证失败：{e}")
        log("token 可能过期或复制错了，重新抓一次。")


def cmd_api(args) -> None:
    """保存控制台开发者 Token（长期有效，首选认证方式）。"""
    print("获取开发者 Token（长期有效，推荐）：")
    print("  1) 浏览器打开 https://www.autodl.com/console/center/settings/token")
    print("  2) 登录后复制 Token（控制台 → 账号 → 设置 → 开发者Token）")
    print("  3) 容器实例 Pro API 需要先完成实名认证；普通实例可能不支持")
    print()
    tok = (args.token_value or "").strip()
    if not tok:
        tok = input("粘贴开发者 Token: ").strip()
    if not tok:
        sys.exit("没输入，放弃。")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(API_TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(tok)
    os.chmod(API_TOKEN_FILE, 0o600)
    log(f"已保存（600）→ {API_TOKEN_FILE}")

    # 验证并判断能走哪套接口
    try:
        r = http_json("POST", API_MAIN + "/api/v1/dev/instance/pro/list",
                      {"page_index": 1, "page_size": 1}, tok)
        if r.get("code") == "Success":
            total = (r.get("data") or {}).get("result_total")
            log(f"✅ 开发者 Token 有效，Pro 接口可用（账号下 {total} 个 Pro 实例）")
            return
        log(f"Pro 接口返回：{r.get('code')} {r.get('msg')}")
    except ApiError as e:
        log(f"Pro 接口调用失败：{e}")
    try:
        r = http_json("POST", API_MAIN + "/api/v1/dev/instance/list",
                      {"page_index": 1, "page_size": 1}, tok)
        if r.get("code") == "Success":
            log("✅ 开发者 Token 可用（普通 dev 接口）")
            return
        log(f"普通 dev 接口返回：{r.get('code')} {r.get('msg')}")
    except ApiError as e:
        log(f"普通 dev 接口调用失败：{e}")
    log("⚠️ Token 存下了，但两套实例接口都调不通——开机仍会用浏览器 token 兜底。")


def cmd_list(args) -> None:
    token, mode = require_auth()
    try:
        ins = list_instances(token, mode)
    except ApiError as e:
        sys.exit(f"获取实例列表失败：{e}")
    if args.json:
        print(json.dumps(ins, ensure_ascii=False, indent=2))
        return
    if not ins:
        print("账号下没有实例。")
        return
    print(f"账号下共 {len(ins)} 个实例：")
    print(f"{'状态':<12}{'uuid':<24}{'名称':<16}{'机器':<8}{'GPU':<16}{'价格':<10}")
    print("-" * 92)
    for x in ins:
        price = x.get("payg_price") or 0
        price_s = f"¥{price / 1000:.2f}/时" if price else "-"
        print(
            f"{str(x.get('status')):<12}"
            f"{str(x.get('uuid')):<24}"
            f"{str(x.get('name') or x.get('instance_name') or '')[:14]:<16}"
            f"{str(x.get('machine_alias') or ''):<8}"
            f"{str(x.get('snapshot_gpu_alias_name') or x.get('gpu_name') or ''):<16}"
            f"{price_s:<10}"
        )
    print("-" * 92)
    print("开机：python3 autodl-grab.py poweron <uuid>")
    print("一键抢：python3 autodl-grab.py grab <uuid>")


def cmd_poweron(args) -> int:
    token, mode = require_auth()
    uuid = args.uuid
    no_notify = args.no_notify

    try:
        ins = find_instance(token, uuid, mode)
    except ApiError as e:
        sys.exit(f"读取实例状态失败：{e}")

    if ins:
        status = ins.get("status")
        start_mode = ins.get("start_mode")
        log(f"实例「{ins.get('name') or ins.get('instance_name') or uuid}」当前状态：{status}"
            + ("（无卡模式）" if start_mode == "non_gpu" else ""))
        if status == "running":
            if start_mode == "non_gpu":
                log("当前是无卡模式在跑，想上有卡请用 `grab` 命令（会先关机再抢）。")
            else:
                log("已经在运行，不需要开机。")
            return 0
    else:
        log(f"⚠️ 实例列表里没找到 {uuid}，仍然尝试开机（可能 uuid 写错或属于子账号）")

    fails = 0
    log(f"开始轮询开机：间隔约 {args.interval}s，Ctrl-C 退出")
    while True:
        try:
            r = power_on(token, uuid, mode)
        except ApiError as e:
            fails += 1
            log(f"请求失败（第 {fails} 次）：{e}")
        else:
            code = r.get("code") or ""
            msg = (r.get("msg") or "").strip()
            if code == "Success":
                log("✅ 开机成功，卡抢到了")
                # 等实例进入 running
                for _ in range(10):
                    time.sleep(3)
                    try:
                        cur = find_instance(token, uuid, mode)
                    except ApiError:
                        break
                    if cur and cur.get("status") == "running":
                        log("实例已进入 running。")
                        ins = cur
                        break
                run_cmd = getattr(args, "run_command", None)
                run_ok = None
                run_out = ""
                if run_cmd:
                    run_ok, run_out = False, ""
                    for attempt in range(4):
                        run_ok, run_out = run_over_ssh(ins, run_cmd)
                        if run_ok:
                            break
                        if attempt < 3:
                            log(f"远程命令第 {attempt + 1} 次失败，10 秒后重试：{run_out[:120]}")
                            time.sleep(10)
                    if run_ok:
                        log("✅ 远程命令已下发，训练应该已经开始")
                        if not no_notify:
                            notify("AutoDL 训练已启动", "抢到卡 + 远程命令执行成功")
                    else:
                        log(f"⚠️ 远程命令失败：{run_out[:300]}")
                        if not no_notify:
                            notify("AutoDL 命令执行失败", run_out[:100])
                if not no_notify:
                    notify("AutoDL 抢卡成功", f"实例已开机：{uuid}")
                    if not getattr(args, "no_mail", False):
                        _mail_success(ins, token, uuid, run_cmd, run_ok, run_out)
                return 0
            if code == "AuthorizeFailed" or "登录超时" in msg:
                sys.exit(f"token 失效：{msg}（运行 token 命令更新）")
            fails += 1
            log(f"未抢到（第 {fails} 次）：{msg or code}")

        if args.max_tries and fails >= args.max_tries:
            log(f"连续失败 {fails} 次，按 --max-tries 放弃。")
            return 1

        time.sleep(args.interval * random.uniform(0.9, 1.1))


def run_over_ssh(ins: dict | None, command: str) -> tuple[bool, str]:
    """通过实例详情里的 ssh_command 在远端执行命令。返回 (成功?, 输出)。"""
    ssh_command = (ins or {}).get("ssh_command") or ""
    if not ssh_command:
        return False, "实例详情里没有 ssh_command"
    parts = ssh_command.split()
    host = parts[-1]
    port = "22"
    if "-p" in parts:
        i = parts.index("-p")
        if i + 1 < len(parts):
            port = parts[i + 1]
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20", "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=15",
        "-p", port, host, command,
    ]
    log(f"SSH → {host}:{port} 执行：{command}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "SSH 执行超时（180s）"
    except Exception as e:
        return False, f"SSH 异常：{e}"
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    return r.returncode == 0, out


def _mail_success(ins: dict | None, token: str, uuid: str,
                  run_command: str | None = None,
                  run_ok: bool | None = None,
                  run_out: str = "") -> None:
    """抢到卡后发邮件通知。"""
    name = (ins or {}).get("name") or (ins or {}).get("instance_name") or uuid
    machine = (ins or {}).get("machine_alias") or ""
    region = (ins or {}).get("region_name") or ""
    gpu = (ins or {}).get("snapshot_gpu_alias_name") or (ins or {}).get("gpu_name") or ""
    price = (ins or {}).get("payg_price") or 0
    price_s = f"¥{price / 1000:.2f}/时" if price else "未知"
    ssh_cmd = (ins or {}).get("ssh_command") or "ssh atri-gpu"
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "抢到卡了，实例已经开机。",
        "",
        f"实例：{name}",
        f"机器：{region} {machine}".strip(),
        f"GPU：{gpu}",
        f"价格：{price_s}",
        f"时间：{now_s}",
        "",
        f"SSH：{ssh_cmd}",
        "",
    ]
    if run_command:
        lines.append("")
        if run_ok:
            lines.append("开机后自动执行的命令：成功")
            lines.append(run_command)
        else:
            lines.append("开机后自动执行的命令：失败")
            lines.append(run_command)
            lines.append("输出：" + (run_out[:500] or "（无输出）"))
    else:
        lines += [
            "",
            "接下来可以上去跑训练了：",
            "cd /root/autodl-tmp/atri && nohup bash run_v6_ab.sh > /root/autodl-tmp/atri-runs/v6-ab-nohup.log 2>&1 &",
        ]
    lines += ["", "（按量计费，跑完记得关机。）"]
    log("正在发邮件通知 ...")
    if send_mail("AutoDL 抢到卡了 · " + str(name), lines):
        log("✅ 邮件已发送")
    else:
        log("⚠️ 邮件没发出去，看上面的错误")


def power_off(token: str, uuid: str, mode: str = "jwt") -> dict:
    if mode == "api-pro":
        return http_json("POST", API_MAIN + "/api/v1/dev/instance/pro/power_off",
                         {"instance_uuid": uuid}, token)
    if mode == "api":
        return http_json("POST", API_MAIN + "/api/v1/dev/instance/power_off",
                         {"instance_uuid": uuid}, token)
    return http_json("POST", API_MAIN + "/api/v1/instance/power_off",
                     {"instance_uuid": uuid}, token)


def cmd_grab(args) -> int:
    """一键抢卡：如果实例在跑（比如无卡模式），先关机，然后死等有卡开机。"""
    token, mode = require_auth()
    uuid = args.uuid

    try:
        ins = find_instance(token, uuid, mode)
    except ApiError as e:
        sys.exit(f"读取实例状态失败：{e}")
    if not ins:
        sys.exit(f"实例列表里没找到 {uuid}。先用 list 命令确认 uuid。")

    status = ins.get("status")
    start_mode = ins.get("start_mode")
    mode_label = "无卡模式" if start_mode == "non_gpu" else "有卡模式"
    log(f"实例「{ins.get('name') or ins.get('instance_name') or uuid}」当前：{status}（{mode_label}）")

    if status == "running":
        print()
        print("要抢有卡，必须先关机（无卡模式也一样），然后马上轮询有卡开机。")
        print("数据盘 /root/autodl-tmp 关机不会丢，可以放心。")
        if not args.yes:
            input("确认关机并开始抢卡？回车继续，Ctrl-C 取消：")
        log("正在关机 ...")
        try:
            r = power_off(token, uuid, mode)
        except ApiError as e:
            sys.exit(f"关机请求失败：{e}")
        if r.get("code") != "Success":
            sys.exit(f"关机失败：{r.get('code')} {r.get('msg')}")

        deadline = time.time() + 120
        while time.time() < deadline:
            time.sleep(3)
            try:
                cur = find_instance(token, uuid, mode)
            except ApiError:
                continue
            st = (cur or {}).get("status")
            if st == "shutdown":
                log("已关机，开始抢卡。")
                break
            log(f"关机中 ... 当前 {st}")
        else:
            sys.exit("等关机超时（120s），先看看网页上的状态。")
    elif status != "shutdown":
        log(f"⚠️ 当前状态 {status} 不是 shutdown，仍然尝试开机。")
    else:
        log("已关机，直接开始抢卡。")

    # 关机后复用 poweron 的轮询逻辑
    return cmd_poweron(args)


def cmd_watch(args) -> int:
    """盯库存：有货响铃；给了 --poweron 就顺手开机。"""
    token, mode = "", "jwt"
    if args.poweron:
        token, mode = require_auth()

    regions = args.region or DEFAULT_REGIONS
    kws = [k.strip().lower() for k in (args.gpu or []) if k.strip()]
    last_idle: dict[str, int] = {}
    log(f"开始盯库存：{', '.join(REGION_NAMES.get(r, r) for r in regions)}"
        f" · 筛选 {', '.join(kws) or '全部'} · 间隔约 {args.interval}s，Ctrl-C 退出")

    while True:
        try:
            rows = query_stock(regions)
        except ApiError as e:
            log(f"查询失败：{e}")
            time.sleep(args.interval)
            continue

        if kws:
            rows = [g for g in rows if any(k in (g.get("user_define_name") or "").lower() for k in kws)]
        rows = [g for g in rows if (g.get("gpu_idle_num") or 0) > 0]

        for g in rows:
            name = g.get("user_define_name") or "?"
            idle = g.get("gpu_idle_num") or 0
            if idle != last_idle.get(name):
                last_idle[name] = idle
                log(f"🟢 {name} 空闲 {idle}/{g.get('gpu_total_number')} 张")
                notify("AutoDL 有卡了", f"{name} 空闲 {idle} 张")
                if args.poweron:
                    log(f"尝试开机 {args.poweron} ...")
                    try:
                        r = power_on(token, args.poweron, mode)
                    except ApiError as e:
                        log(f"开机请求失败：{e}")
                    else:
                        if r.get("code") == "Success":
                            log("✅ 自动开机成功")
                            notify("AutoDL 抢卡成功", f"实例已开机：{args.poweron}")
                            return 0
                        log(f"开机未成功：{r.get('code')} {r.get('msg')}")
        time.sleep(args.interval * random.uniform(0.85, 1.15))


# ────────────────────────────────────────────── 入口 ──

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AutoDL 抢卡 / 开机小工具（库存查询走匿名接口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""例子:
  python3 autodl-grab.py stock --gpu 5090
  python3 autodl-grab.py token
  python3 autodl-grab.py list
  python3 autodl-grab.py poweron <uuid>
  python3 autodl-grab.py watch --gpu 5090 --poweron <uuid>
""")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stock", help="查实时库存（匿名）")
    s.add_argument("--region", nargs="+", default=None, help="区域 id，默认 westDC3 beijingDC2")
    s.add_argument("--gpu", nargs="+", default=None, help="型号关键字过滤，如 5090 4090 H800")
    s.add_argument("--json", action="store_true", help="输出原始 JSON")
    s.set_defaults(func=cmd_stock)

    s = sub.add_parser("token", help="保存浏览器 Authorization token")
    s.add_argument("--token", dest="token_value", default=None, help="直接传 token（不交互）")
    s.set_defaults(func=cmd_token)

    s = sub.add_parser("api", help="保存开发者 Token（首选，长期有效）")
    s.add_argument("--token", dest="token_value", default=None, help="直接传 token（不交互）")
    s.set_defaults(func=cmd_api)

    s = sub.add_parser("list", help="列出账号下的实例")
    s.add_argument("--json", action="store_true", help="输出原始 JSON")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("poweron", help="轮询开机，抢到为止")
    s.add_argument("uuid", help="实例 uuid")
    s.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="轮询间隔秒，默认 5")
    s.add_argument("--max-tries", type=int, default=0, help="连续失败多少次后放弃，0=无限")
    s.add_argument("--no-notify", action="store_true", help="成功后不弹通知")
    s.add_argument("--no-mail", action="store_true", help="成功后不发邮件")
    s.add_argument("--run", dest="run_command", default=None,
                   help="开机成功后自动在服务器执行的命令")
    s.set_defaults(func=cmd_poweron)

    s = sub.add_parser("grab", help="一键抢卡：先关机（如需要），再死等有卡开机")
    s.add_argument("uuid", help="实例 uuid")
    s.add_argument("--yes", action="store_true", help="不询问，直接关机")
    s.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="轮询间隔秒，默认 5")
    s.add_argument("--max-tries", type=int, default=0, help="连续失败多少次后放弃，0=无限")
    s.add_argument("--no-notify", action="store_true", help="成功后不弹通知")
    s.add_argument("--no-mail", action="store_true", help="成功后不发邮件")
    s.add_argument("--run", dest="run_command", default=None,
                   help="开机成功后自动在服务器执行的命令")
    s.set_defaults(func=cmd_grab)

    s = sub.add_parser("watch", help="盯库存，有货响铃（可自动开机）")
    s.add_argument("--region", nargs="+", default=None, help="区域 id，默认 westDC3 beijingDC2")
    s.add_argument("--gpu", nargs="+", default=None, help="型号关键字过滤")
    s.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="轮询间隔秒，默认 5")
    s.add_argument("--poweron", default=None, help="有货时自动开机指定 uuid")
    s.set_defaults(func=cmd_watch)

    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = args.func(args)
        return result if isinstance(result, int) else 0
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
