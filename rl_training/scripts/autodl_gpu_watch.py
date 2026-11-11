#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AutoDL 西北B区 RTX 4090 盯卡器（非官方 Web API，仅操作自己账号）。

背景：实例关机后 GPU 不预留，原主机被占满时开机报 MachineGpuNumUpdateFailed，
唯一出路是同区「克隆实例（拷贝数据盘）」到有卡主机。本脚本：
  1. 轮询地区级空闲卡数（POST /api/v1/machine/region/gpu_type）；
  2. 见卡后优先「原主机直接开机」（POST /api/v1/instance/power_on, payload=gpu）；
  3. 原主机没卡则克隆到兼容主机（POST /api/v1/order/instance/clone/payg，
     copy_data_disk_after_clone=true，下单前写 pending 闩锁防重复下单）；
  4. 跟踪到 running + 数据盘拷贝 Succeed，把 SSH 信息写进结果文件后退出。

接口与校验逻辑参考社区开源实现（autodl-autograb / card-helper-pro / GPUHawk 等，
多源交叉证实），非官方承诺，随时可能失效。凭据 = 网页登录 JWT（Authorization 头，
约 60 天有效），只从 --token-file 或环境变量 AUTODL_TOKEN 读取，永不写日志/入库。

用法：
  python autodl_gpu_watch.py probe                     # 免登录连通性自检
  python autodl_gpu_watch.py discover --token-file F   # 列实例、定位源实例与快照
  python autodl_gpu_watch.py check    --token-file F   # 单次库存查询
  python autodl_gpu_watch.py watch    --token-file F   # 盯卡循环（见卡自动开机/克隆）

安全边界：只做 power_on / clone / 查询；绝不调用 power_off / release；
余额不足(TORecharge)、未实名(TORealName)、token 失效等致命码 → 立即退出报警。
"""
import argparse
import json
import math
import os
import random
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid as uuidlib
from datetime import datetime

SITE = 'https://www.autodl.com'
REGION_URL = 'https://fe-config-backend.autodl.com/api/v1/autodl/region/tag'
GIB = 1024 ** 3
STATE_PATH = r'd:\tmp\autodl_watch_state.json'
RESULT_PATH = r'd:\tmp\autodl_watch_result.json'
LOG_PATH = r'd:\tmp\autodl_watch.log'
REGION_NAME = '西北B区'
FALLBACK_SIGNS = ['west-B', 'west-C', 'west-D', 'west-E']
GPU_MODEL = 'RTX 4090'
FATAL_CODES = {'TORecharge', 'TORealName', 'AccountChanged'}
TOKEN_FATAL = ('AuthorizeFailed', 'Unauthorized', 'TokenExpired', 'TOLogin')

_UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
       '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')


class StopError(Exception):
    """致命错误：退出等人处理。"""


class RetryError(Exception):
    """可重试错误：退避后继续。"""

    def __init__(self, msg, delay=0.0):
        super().__init__(msg)
        self.delay = delay


def log(msg):
    line = '[%s] %s' % (datetime.now().strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write('[%s] %s\n' % (datetime.now().isoformat(timespec='seconds'), msg))
    except OSError:
        pass


def clean(v):
    return ''.join(str(v or '').split()).lower()


def load_token(args):
    tok = os.environ.get('AUTODL_TOKEN', '')
    if getattr(args, 'token_file', None) and os.path.exists(args.token_file):
        with open(args.token_file, encoding='utf-8') as f:
            tok = f.read()
    tok = (tok or '').strip().strip('"').strip("'")
    if not tok.startswith('eyJ'):
        raise StopError('token 缺失或格式不对（应以 eyJ 开头的裸 JWT；'
                        '浏览器 F12 Console 执行 copy(localStorage.token) 获取）')
    return tok


class Api:
    def __init__(self, token=None):
        self.token = token
        self.ctx = ssl.create_default_context()

    def call(self, path, payload=None, timeout=15, creation=False, public=False):
        url = (REGION_URL if public else SITE + path)
        headers = {'Accept': 'application/json', 'User-Agent': _UA}
        data = None
        if not public:
            headers.update({
                'Authorization': self.token,
                'Content-Type': 'application/json',
                'Origin': SITE,
                'Referer': SITE + '/market/list',
            })
            data = json.dumps(payload if payload is not None else {}).encode()
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method='POST' if data else 'GET')
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
                body = json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise StopError('HTTP %d：登录态失效或触发账号验证，需要人工重新拿 token' % e.code)
            if e.code == 429 or e.code >= 500:
                ra = e.headers.get('Retry-After') if e.headers else None
                try:
                    delay = float(ra) if ra else 0.0
                except ValueError:
                    delay = 0.0
                if creation:
                    raise StopError('下单 HTTP %d，结果未知——保留 pending 闩锁，去控制台核对！' % e.code)
                raise RetryError('HTTP %d，退避重试' % e.code, delay)
            raise StopError('HTTP %d @ %s' % (e.code, path))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if creation:
                raise StopError('下单网络异常，结果未知——保留 pending 闩锁，去控制台核对！(%s)' % e)
            raise RetryError('网络/解析错误：%s' % e)
        code = str(body.get('code', ''))
        if any(t.lower() in code.lower() for t in TOKEN_FATAL):
            raise StopError('token 已失效（code=%s），请重新从浏览器复制' % code)
        if code in FATAL_CODES:
            raise StopError('致命业务码 %s：%s' % (code, body.get('msg', '')))
        if creation:
            return body
        if code not in ('Success', 'OK'):
            raise StopError('接口 %s 返回 %s：%s' % (path, code, str(body.get('msg', ''))[:200]))
        return body


# ---------- 地区 / 库存 ----------

def resolve_signs(api):
    try:
        body = api.call('', public=True)
        rows = [r for r in body.get('data', [])
                if clean(r.get('region_name')) == clean(REGION_NAME)
                and r.get('region_class') == 'normal']
        if len(rows) == 1 and isinstance(rows[0].get('region_sign'), list) and rows[0]['region_sign']:
            return sorted(set(rows[0]['region_sign']))
        signs = sorted({s for r in rows for s in (r.get('region_sign') or [])
                        if isinstance(s, str)})
        if signs:
            return signs
    except (StopError, RetryError) as e:
        log('地区目录解析失败（%s），用兜底 signs' % e)
    return FALLBACK_SIGNS


def region_idle(api, signs):
    body = api.call('/api/v1/machine/region/gpu_type', {'region_sign_list': signs})
    idle = 0
    for row in body.get('data') or []:
        for gpu, info in (row or {}).items():
            if clean(gpu) == clean(GPU_MODEL) and isinstance(info, dict):
                idle += int(info.get('idle_gpu_num') or 0)
    return idle


def machine_query(signs, page=1):
    return {'charge_type': 'payg', 'region_sign': '', 'region_sign_list': signs,
            'gpu_type_name': [GPU_MODEL], 'gpu_idle_num': 1, 'default_order': True,
            'machine_tag_name': [], 'mount_net_disk': False, 'cpu_arch': [],
            'chip_corp': [], 'page_index': page, 'page_size': 100}


def list_machines(api, signs):
    out, scanned = [], 0
    for page in range(1, 6):
        data = api.call('/api/v1/user/machine/list', machine_query(signs, page)).get('data') or {}
        rows = data.get('list') or []
        out += rows
        scanned += len(rows)
        total = data.get('result_total')
        if not rows or (isinstance(total, (int, float)) and scanned >= total):
            break
    return out


def cuda_version(v):
    t = str(v or '').strip().lower().replace('cuda', '').strip()
    try:
        if '.' in t:
            a, b = t.split('.')[:2]
            return int(a) * 100 + int(b or 0)
        if len(t) == 3 and t.isdigit():
            return int(t[:2]) * 100 + int(t[2])
        if t.isdigit():
            return int(t) * 100
    except ValueError:
        pass
    return -1


def candidates(api, signs, source, expand_bytes, max_gpu_cny):
    good = []
    for m in list_machines(api, signs):
        if clean(m.get('gpu_name')) != clean(GPU_MODEL):
            continue
        if m.get('region_sign') not in signs:
            continue
        if not isinstance(m.get('gpu_idle_num'), (int, float)) or m['gpu_idle_num'] < 1:
            continue
        if m.get('on_line') in (3, 4, 5, 6):
            continue
        mi, bi = m.get('max_instance_num'), m.get('binding_instance_num')
        if not (isinstance(mi, (int, float)) and isinstance(bi, (int, float)) and mi > bi):
            continue
        mdes = m.get('max_data_disk_expand_size')
        if isinstance(mdes, (int, float)) and mdes < expand_bytes:
            continue
        if source and m.get('machine_id') == source['machine_id']:
            continue  # 原主机走 power_on 路径，不做克隆目标
        if source and (m.get('cpu_arch') != source['cpu_arch']
                       or m.get('chip_corp') != source['chip_corp']
                       or cuda_version(m.get('highest_cuda_version')) < source['cuda_num']):
            continue
        price = m.get('payg_price')
        if isinstance(price, (int, float)) and price / 1000.0 > max_gpu_cny:
            log('跳过 %s：报价 ¥%.2f/h 超上限' % (m.get('machine_id'), price / 1000.0))
            continue
        good.append(m)
    return good


# ---------- 实例 / 源快照 ----------

def list_instances(api, name=''):
    body = api.call('/api/v1/instance', {'page_index': 1, 'page_size': 100,
                                          'status': [], 'charge_type': [], 'name': name})
    return (body.get('data') or {}).get('list') or []


def find_source(api, signs, want_uuid=None):
    rows = list_instances(api)
    matches = []
    for it in rows:
        u = it.get('instance_uuid', '')
        if want_uuid and u != want_uuid:
            continue
        gpu = str(it.get('gpu_name') or it.get('gpu_type') or '')
        region = it.get('region_sign') or ''
        if (not want_uuid) and (clean(gpu) != clean(GPU_MODEL) or region not in signs):
            continue
        matches.append(it)
    if not matches:
        with open(r'd:\tmp\autodl_instances_dump.json', 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        raise StopError('实例列表里找不到%s的 %s 实例（共 %d 条，已 dump 到 d:\\tmp\\autodl_instances_dump.json 人工看）'
                        % (want_uuid or (' %s 区' % REGION_NAME), GPU_MODEL, len(rows)))
    if len(matches) > 1 and not want_uuid:
        raise StopError('匹配到 %d 个实例，无法自动选定——请用 --instance 指定：%s'
                        % (len(matches), [m.get('instance_uuid') for m in matches]))
    return matches[0]


def dump_instances(api):
    rows = list_instances(api)
    with open(r'd:\tmp\autodl_instances_dump.json', 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return rows


def get_source_info(api, inst):
    u = inst['instance_uuid']
    detail = (api.call('/api/v1/instance/detail?instance_uuid=' + u).get('data')) or {}
    snap = (api.call('/api/v1/instance/snapshot?instance_uuid=' + u).get('data')) or {}
    image = snap.get('image') or {}
    if image.get('image_type') not in (None, 'public'):
        raise StopError('源实例镜像类型=%s 非 public，克隆 payload 未验证——请走控制台人工克隆'
                        % image.get('image_type'))
    expand = snap.get('expand_data_disk_size')
    expand = int(expand) if isinstance(expand, (int, float)) else 0
    src = {
        'uuid': u,
        'name': detail.get('instance_name') or inst.get('instance_name') or '',
        'status': detail.get('status') or inst.get('status') or '',
        'machine_id': detail.get('machine_id') or u.split('-')[0],
        'region_sign': (snap.get('machine_info_snapshot') or {}).get('region_sign')
                       or detail.get('region_sign') or inst.get('region_sign') or '',
        'base_image_uuid': image.get('base_image_uuid') or '',
        'cuda': str(image.get('cuda_version') or ''),
        'cuda_num': cuda_version(image.get('cuda_version')),
        'cpu_arch': image.get('cpu_arch') or (snap.get('cpu_arch') or ''),
        'chip_corp': image.get('chip_corp') or (snap.get('chip_corp') or ''),
        'expand_bytes': expand,
        'req_gpu_amount': detail.get('req_gpu_amount'),
    }
    return src, detail, snap


# ---------- 状态闩锁（防重复下单） ----------

def load_state():
    try:
        with open(STATE_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_state(st):
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


def clear_state():
    try:
        os.remove(STATE_PATH)
    except OSError:
        pass


# ---------- 动作 ----------

def wait_running(api, u, timeout, need_gpu=True):
    t0 = time.time()
    last = ''
    while time.time() - t0 < timeout:
        try:
            for it in list_instances(api, name=u):
                if it.get('instance_uuid') != u:
                    continue
                st = it.get('status', '')
                if st != last:
                    log('实例 %s 状态：%s -> %s' % (u, last or '(初始)', st))
                    last = st
                if st == 'running' and (not need_gpu or it.get('start_mode') in (None, 'gpu')):
                    return it
                if st in ('create_failed', 're_init_failed'):
                    log('状态 %s，尝试 init 修复' % st)
                    api.call('/api/v1/instance/init', {'instance_uuid': u}, creation=True)
                if st == 'created_clone_failed':
                    log('克隆失败态，尝试 clone/retry 一次')
                    api.call('/api/v1/instance/clone/retry', {'instance_uuid': u}, creation=True)
        except RetryError as e:
            log('查询实例状态可重试错误：%s' % e)
        time.sleep(10)
    raise StopError('等待实例 %s 变 running 超时（%ds），去控制台看' % (u, timeout))


def wait_data_copy(api, u, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            for it in list_instances(api, name=u):
                if it.get('instance_uuid') != u:
                    continue
                cs = ((it.get('additional_info') or {}).get('copy_data_disk_after_clone_status')
                      or it.get('copy_data_disk_after_clone_status') or '')
                if cs in ('Succeed', 'Success'):
                    return True
                if cs and 'fail' in cs.lower():
                    raise StopError('数据盘拷贝失败（%s）——用控制台「跨实例拷贝数据」补救' % cs)
        except RetryError as e:
            log('查询拷贝状态可重试错误：%s' % e)
        time.sleep(20)
    log('数据盘拷贝 %ds 还没到 Succeed——不再等，人工核对' % timeout)
    return False


def do_power_on(api, src):
    log('原主机 %s 有空闲卡 → 直接开机 %s（payload=gpu）' % (src['machine_id'], src['uuid']))
    body = api.call('/api/v1/instance/power_on',
                    {'instance_uuid': src['uuid'], 'payload': 'gpu'}, creation=True)
    code = str(body.get('code', ''))
    if code in ('MachineGpuNumUpdateFailed', 'GpuUnavailable'):
        log('开机被拒（%s）：卡瞬间被抢/主机满 → 转克隆路线' % code)
        return None
    if code != 'Success':
        raise StopError('power_on 返回 %s：%s——去控制台核对' % (code, str(body.get('msg', ''))[:200]))
    it = wait_running(api, src['uuid'], 300)
    return {'mode': 'power_on', 'instance_uuid': src['uuid'], 'item': it}


def do_clone(api, src, signs, args):
    srcs = candidates(api, signs, src, src['expand_bytes'], args.max_gpu_cny)
    if not srcs:
        return None
    m = srcs[0]
    mid = m['machine_id']
    log('克隆目标主机 %s（%s，空闲 %s 卡，¥%.2f/h）'
        % (mid, m.get('region_sign'), m.get('gpu_idle_num'),
           (m.get('payg_price') or 0) / 1000.0))
    # 下单前最后核对：源实例仍关机、配置未变
    fresh_inst = find_source(api, signs, want_uuid=src['uuid'])
    fresh, fdetail, _ = get_source_info(api, fresh_inst)
    if fresh['status'] != 'shutdown':
        raise StopError('源实例状态=%s 非 shutdown，克隆前置不满足' % fresh['status'])
    for k in ('machine_id', 'base_image_uuid', 'cuda', 'cpu_arch', 'chip_corp', 'expand_bytes'):
        if fresh[k] != src[k]:
            raise StopError('源实例配置在盯卡期间变了（%s），停下人工核对' % k)
    attempt = uuidlib.uuid4().hex[:8]
    name = 'rl-clone-%s' % attempt
    expand_gb = int(math.ceil(src['expand_bytes'] / GIB)) if src['expand_bytes'] else 0
    payload = {
        'instance_uuid': src['uuid'],
        'instance_info': {
            'machine_id': mid, 'charge_type': 'payg', 'req_gpu_amount': 1,
            'base_image_uuid': src['base_image_uuid'], 'instance_name': name,
            'private_image_uuid': '', 'reproduction_uuid': '', 'cg_application_uuid': '',
            'cg_application_info': {}, 'expand_data_disk': expand_gb * GIB,
            'copy_data_disk_after_clone': True,
            'keep_src_user_service_address_after_clone': False,
            'reproduction_id': 0, 'ft_sparse': True,
        },
        'price_info': {'coupon_id_list': [], 'machine_id': mid, 'charge_type': 'payg',
                       'duration': 1, 'num': 1, 'expand_data_disk': expand_gb * GIB},
    }
    # 持久闩锁先落盘，再发计费 POST —— 网络中断也不会二次下单
    save_state({'status': 'pending', 'op': 'clone', 'attempt': attempt,
                'source_uuid': src['uuid'], 'machine_id': mid, 'name': name,
                'at': datetime.now().isoformat(timespec='seconds')})
    body = api.call('/api/v1/order/instance/clone/payg', payload, timeout=60, creation=True)
    code = str(body.get('code', ''))
    data = body.get('data')
    if code == 'Success' and isinstance(data, str) and data:
        save_state({'status': 'accepted', 'op': 'clone', 'attempt': attempt,
                    'source_uuid': src['uuid'], 'new_uuid': data, 'machine_id': mid,
                    'name': name, 'at': datetime.now().isoformat(timespec='seconds')})
        log('克隆下单成功，新实例 %s，数据盘开始拷贝' % data)
        it = wait_running(api, data, 900)
        if it.get('status') == 'shutdown':  # 极少数创建后未自动开机
            api.call('/api/v1/instance/power_on', {'instance_uuid': data, 'payload': 'gpu'},
                     creation=True)
            it = wait_running(api, data, 300)
        copied = wait_data_copy(api, data, 2400)
        return {'mode': 'clone', 'instance_uuid': data, 'machine_id': mid,
                'data_disk_copied': copied, 'item': it}
    if code == 'MachineGpuNumUpdateFailed':
        clear_state()
        log('克隆被拒：目标机卡片瞬间被抢，回到盯卡')
        return None
    raise StopError('克隆下单返回 %s：%s——pending 闩锁已保留，去控制台核对是否已产生新实例！'
                    % (code, str(body.get('msg', ''))[:200]))


def resume_if_pending(api, signs):
    st = load_state()
    if not st:
        return None
    if st.get('status') == 'accepted' and st.get('new_uuid'):
        log('发现 accepted 闩锁（新实例 %s）——续跟踪，不重复下单' % st['new_uuid'])
        it = wait_running(api, st['new_uuid'], 900)
        copied = wait_data_copy(api, st['new_uuid'], 2400)
        return {'mode': 'clone(resumed)', 'instance_uuid': st['new_uuid'],
                'machine_id': st.get('machine_id'), 'data_disk_copied': copied, 'item': it}
    if st.get('status') == 'pending':
        raise StopError('上次下单结果未知（pending 闩锁在 %s）——先去控制台核对有没有多出来的新实例；'
                        '确认没有后删除该文件再重跑' % STATE_PATH)
    return None


def finish(result):
    item = result.get('item') or {}
    ssh = {k: v for k, v in item.items()
           if any(s in k.lower() for s in ('ssh', 'host', 'port', 'password', 'address'))}
    out = {'ok': True, 'mode': result['mode'], 'instance_uuid': result['instance_uuid'],
           'machine_id': result.get('machine_id'),
           'data_disk_copied': result.get('data_disk_copied'),
           'ssh_info': ssh, 'raw_item': item,
           'at': datetime.now().isoformat(timespec='seconds')}
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    save_state({'status': 'done', 'instance_uuid': result['instance_uuid'],
                'mode': result['mode'], 'at': out['at']})
    log('=' * 60)
    log('成功！mode=%s 实例=%s SSH信息已写 %s' % (result['mode'], result['instance_uuid'], RESULT_PATH))
    log('=' * 60)


# ---------- 模式 ----------

def mode_probe(args):
    api = Api()
    body = api.call('', public=True)
    n = len(body.get('data') or [])
    log('probe OK：地区目录 %d 条' % n)
    signs = resolve_signs(api)
    log('%s region_sign = %s' % (REGION_NAME, signs))
    pro = json.loads(urllib.request.urlopen(urllib.request.Request(
        SITE + '/api/v1/machine_pro/gpu_type',
        data=json.dumps({'data_center_list': ['westDC3']}).encode(),
        headers={'Content-Type': 'application/json', 'User-Agent': _UA}),
        timeout=15, context=api.ctx).read().decode())
    log('Pro 池（免登录，仅辅助信号，无普通 4090）：%s'
        % json.dumps(pro.get('data'), ensure_ascii=False)[:300])


def mode_discover(args):
    api = Api(load_token(args))
    signs = resolve_signs(api)
    log('%s signs=%s' % (REGION_NAME, signs))
    rows = dump_instances(api)
    log('实例列表已 dump 到 d:\\tmp\\autodl_instances_dump.json（%d 条）' % len(rows))
    src_inst = find_source(api, signs, want_uuid=args.instance)
    src, detail, snap = get_source_info(api, src_inst)
    log('源实例：%s' % json.dumps(src, ensure_ascii=False, indent=1))
    log('状态=%s 主机=%s 卡数=%s 数据盘扩容=%sB' %
        (src['status'], src['machine_id'], src['req_gpu_amount'], src['expand_bytes']))
    idle = region_idle(api, signs)
    log('当前 %s %s 空闲合计：%d' % (REGION_NAME, GPU_MODEL, idle))


def mode_check(args):
    api = Api(load_token(args))
    signs = resolve_signs(api)
    idle = region_idle(api, signs)
    log('%s %s 空闲合计：%d' % (REGION_NAME, GPU_MODEL, idle))
    if idle:
        for m in candidates(api, signs, None, 0, args.max_gpu_cny):
            log('  有卡主机 %s（%s）空闲 %s，¥%.2f/h，on_line=%s'
                % (m.get('machine_id'), m.get('region_sign'), m.get('gpu_idle_num'),
                   (m.get('payg_price') or 0) / 1000.0, m.get('on_line')))


def mode_watch(args):
    api = Api(load_token(args))
    signs = resolve_signs(api)
    src_inst = find_source(api, signs, want_uuid=args.instance)
    src, detail, _ = get_source_info(api, src_inst)
    log('盯卡启动：源实例 %s（%s，主机 %s，状态 %s）；%s signs=%s'
        % (src['uuid'], src['name'], src['machine_id'], src['status'], REGION_NAME, signs))
    if src['status'] == 'running':
        log('源实例已经在跑了，不用盯，退出')
        return
    if src['status'] != 'shutdown':
        raise StopError('源实例状态=%s（需要 shutdown 才能走本脚本）' % src['status'])
    resumed = resume_if_pending(api, signs)
    if resumed:
        finish(resumed)
        clear_state()
        return
    fails = 0
    poll = 0
    while True:
        poll += 1
        t0 = time.time()
        try:
            idle = region_idle(api, signs)
            if idle < 1:
                if poll % 10 == 1:
                    log('第 %d 轮：无空闲卡（下轮 %ds 后）' % (poll, args.interval))
                fails = 0
            else:
                log('第 %d 轮：发现 %d 张空闲 %s！' % (poll, idle, GPU_MODEL))
                # 路线1：原主机直接开机
                result = None
                try:
                    for m in list_machines(api, signs):
                        if m.get('machine_id') == src['machine_id'] and \
                                (m.get('gpu_idle_num') or 0) >= 1 and m.get('on_line') not in (3, 4, 5, 6):
                            result = do_power_on(api, src)
                            break
                except RetryError as e:
                    log('power_on 路径可重试错误：%s' % e)
                # 路线2：克隆到有卡主机
                if result is None:
                    try:
                        result = do_clone(api, src, signs, args)
                    except RetryError as e:
                        log('克隆路径可重试错误：%s' % e)
                if result:
                    finish(result)
                    return
                fails = 0
        except RetryError as e:
            fails += 1
            backoff = max(e.delay, min(300, args.interval * (2 ** min(fails, 4))))
            log('可重试错误（连续 %d 次）：%s → %.0fs 后再试' % (fails, e, backoff))
            time.sleep(backoff + random.uniform(0, 3))
            continue
        time.sleep(max(5.0, args.interval * random.uniform(0.8, 1.3) - (time.time() - t0)))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='AutoDL 西北B 4090 盯卡器')
    ap.add_argument('mode', choices=['probe', 'discover', 'check', 'watch'])
    ap.add_argument('--token-file', default=os.path.expanduser(r'~\.autodl_token.txt'))
    ap.add_argument('--instance', default=None, help='源实例 uuid（缺省自动匹配唯一的西北B 4090）')
    ap.add_argument('--interval', type=float, default=20.0, help='轮询秒数（默认20，带抖动）')
    ap.add_argument('--max-gpu-cny', type=float, default=2.6, help='GPU 每小时报价上限（元）')
    args = ap.parse_args()
    try:
        {'probe': mode_probe, 'discover': mode_discover,
         'check': mode_check, 'watch': mode_watch}[args.mode](args)
    except StopError as e:
        log('【停止】%s' % e)
        sys.exit(2)
    except KeyboardInterrupt:
        log('手动中断')
        sys.exit(130)


if __name__ == '__main__':
    main()
