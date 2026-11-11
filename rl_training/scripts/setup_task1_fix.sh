#!/usr/bin/env bash
# Task 1 环境初始化·修复加速版（setup_task1.sh STAGE2 断流后的替代方案）
#
# 思路（实测驱动）：
#   - defects4j.org 直连可达（884KB/s 单连接），repos zip 实测仅 602MB
#   - project_repos 不是逐仓库 GitHub clone，而是 defects4j.org 的单个 zip
#   - init.sh/get_repos.sh 的 download_url 条件 GET 失败会 rm -f 重下 → 预置大文件有被删风险
# 方案：aria2c 16 连接预下载全部 8 个工件（大文件直连 / GitHub 小件走学术加速）
#       → 手动解压 repos+gradle → 给两处 download_url 打 local-first 短路补丁
#       → 断网态跑 init.sh（全程不再触网）
#
# 前置：/root/defects4j 已浅克隆（344M）；STAGE1 apt 依赖已装
# 日志：/root/setup_fix.log ；标记：FIX_DONE / FIX<N>_FAIL
LOG=/root/setup_fix.log
exec >"$LOG" 2>&1
set -x
D4J=/root/defects4j
ts(){ date '+%F %T'; }

echo "=== [$(ts)] FIX 1: apt aria2 + perl deps（cpanfile 8 个依赖全有 deb 包）==="
apt-get install -y aria2 libdbd-csv-perl libdbi-perl libjson-perl libjson-parse-perl \
  libperl-critic-perl libstring-interpolate-perl liburi-perl || { echo FIX1_FAIL; exit 31; }
perl -e 'use DBI; use DBD::CSV; use JSON; use JSON::Parse; use URI; print "PERL_DEPS_OK\n"' \
  || { echo FIX1_FAIL_PERL; exit 32; }

# dl <url> <destdir> <outname> <proxy|direct>
dl(){
  local url="$1" dest="$2" out="$3" mode="$4"
  mkdir -p "$dest" || return 1
  cd "$dest" || return 1
  if [ "$mode" = proxy ]; then source /etc/network_turbo >/dev/null 2>&1
  else unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; fi
  aria2c -x16 -s16 -c --file-allocation=none --summary-interval=30 --console-log-level=warn -o "$out" "$url" \
    || aria2c -x16 -s16 -c --file-allocation=none --console-log-level=warn -o "$out" "$url"
}

echo "=== [$(ts)] FIX 2: 预下载 8 个工件 ==="
B=https://defects4j.org/downloads
dl $B/defects4j-repos-v3.zip $D4J/project_repos defects4j-repos-v3.zip direct || { echo FIX2_FAIL_repos; exit 33; }
sz=$(stat -c %s $D4J/project_repos/defects4j-repos-v3.zip)
[ "$sz" = 631727844 ] || { echo "FIX2_BAD_SIZE_repos=$sz (expect 631727844)"; exit 33; }
dl $B/defects4j-gradle-dists-v3.zip $D4J/framework/lib/build_systems/gradle defects4j-gradle-dists-v3.zip direct || { echo FIX2_FAIL_gdists; exit 33; }
dl $B/defects4j-gradle-deps-v3.zip $D4J/framework/lib/build_systems/gradle defects4j-gradle-deps-v3.zip direct || { echo FIX2_FAIL_gdeps; exit 33; }
dl https://mutation-testing.org/downloads/major-3.0.1_jre11.zip $D4J major-3.0.1_jre11.zip direct || { echo FIX2_FAIL_major; exit 33; }
G=https://github.com
dl $G/EvoSuite/evosuite/releases/download/v1.1.0/evosuite-1.1.0.jar $D4J/framework/lib/test_generation/generation evosuite-1.1.0.jar proxy || { echo FIX2_FAIL_evosuite; exit 33; }
dl $G/EvoSuite/evosuite/releases/download/v1.1.0/evosuite-standalone-runtime-1.1.0.jar $D4J/framework/lib/test_generation/runtime evosuite-standalone-runtime-1.1.0.jar proxy || { echo FIX2_FAIL_evort; exit 33; }
dl $G/randoop/randoop/releases/download/v4.3.4/randoop-4.3.4.zip $D4J/framework/lib/test_generation/generation randoop-4.3.4.zip proxy || { echo FIX2_FAIL_randoop; exit 33; }
dl $G/jose/build-analyzer/releases/download/v0.0.1/build-analyzer-0.0.1.jar $D4J/framework/lib build-analyzer-0.0.1.jar proxy || { echo FIX2_FAIL_analyzer; exit 33; }

echo "=== [$(ts)] FIX 3: 手动解压 repos + gradle（模仿 get_repos.sh 的 mv 逻辑）==="
cd $D4J/project_repos || exit 34
if [ ! -d commons-lang.git ]; then
  unzip -q defects4j-repos-v3.zip || { echo FIX3_FAIL_unzip; exit 34; }
  mv defects4j/project_repos/* . && rm -rf defects4j || { echo FIX3_FAIL_mv; exit 34; }
fi
ls | head -30
cd $D4J/framework/lib/build_systems/gradle || exit 34
mkdir -p dists
unzip -q -u defects4j-gradle-dists-v3.zip -d dists || { echo FIX3_FAIL_gunzip1; exit 34; }
unzip -q -u defects4j-gradle-deps-v3.zip || { echo FIX3_FAIL_gunzip2; exit 34; }

echo "=== [$(ts)] FIX 4: download_url 打 local-first 补丁 + 断网态 init.sh ==="
/root/miniconda3/bin/python - <<'PY'
patches = [
    ('/root/defects4j/init.sh',
     'download_url() {\n',
     'download_url() {\n    local _bn0; _bn0="$(basename "${1:-none}")"; if [ -f "$_bn0" ]; then echo "LOCAL_SKIP $_bn0"; return 0; fi\n'),
    ('/root/defects4j/project_repos/get_repos.sh',
     'download_url() {\n',
     'download_url() {\n    local _bn0; _bn0="$(basename "${@: -1}")"; if [ -f "$_bn0" ]; then echo "LOCAL_SKIP $_bn0"; return 0; fi\n'),
]
for path, anchor, guard in patches:
    s = open(path).read()
    if 'LOCAL_SKIP' in s:
        print('already patched:', path); continue
    assert anchor in s, f'anchor missing in {path}'
    open(path, 'w').write(s.replace(anchor, guard, 1))
    print('patched:', path)
PY
[ $? -eq 0 ] || { echo FIX4_FAIL_patch; exit 35; }
cd $D4J || exit 35
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
./init.sh || { echo FIX4_FAIL_init; exit 35; }

echo "=== [$(ts)] FIX 5: 验证 ==="
grep -q 'defects4j/framework/bin' /root/.bashrc || echo 'export PATH=$HOME/defects4j/framework/bin:$PATH' >> /root/.bashrc
export PATH=$D4J/framework/bin:$PATH
defects4j -v
ls $D4J/project_repos | head -30
ls $D4J/major/bin 2>/dev/null | head -5
ls $D4J/framework/lib/test_generation/generation/ | head -10
du -sh $D4J; df -h / | tail -1
echo "=== [$(ts)] FIX_DONE ==="
