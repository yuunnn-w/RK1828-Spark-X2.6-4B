#!/usr/bin/env python3
"""汇总 eval_llm.py 的输出 JSON：正确率、崩坏数、未判定数、时延与 token 统计。

用法: python3 eval_summary.py <results.json> [more.json ...]
"""
import json
import statistics
import sys


def summarize(path):
    d = json.load(open(path, encoding='utf-8'))
    res = d.get('results', [])
    stats = d.get('stats', {})
    tot = {'ok': 0, 'bad': 0, 'un': 0, 'deg': 0}
    secs, toks, outs = [], [], []
    worst_rep = []
    for r in res:
        v = r.get('verdict')
        if v in ('exact', 'numeric', 'letter'):
            tot['ok'] += 1
        elif v == 'unscored':
            tot['un'] += 1
        else:
            tot['bad'] += 1
        if r.get('degen', 0) > 0.6:
            tot['deg'] += 1
        if r.get('sec') is not None:
            secs.append(r['sec'])
        u = r.get('usage') or {}
        if u.get('completion_tokens'):
            toks.append(u['completion_tokens'])
        if u.get('prompt_tokens'):
            outs.append(u['prompt_tokens'])
        worst_rep.append((r.get('degen', 0), r.get('i'), r.get('domain'), (r.get('got_line') or '')[:50]))
    print('== %s ==' % path)
    print('   tag=%s model=%s max_tokens=%s temperature=%s' %
          (d.get('tag'), d.get('model'), d.get('max_tokens'), d.get('temperature')))
    print('   条目 %d | 正确 %d | 错误 %d | 未判定 %d | 复读崩坏 %d' %
          (len(res), tot['ok'], tot['bad'], tot['un'], tot['deg']))
    scored = tot['ok'] + tot['bad']
    if scored:
        print('   可判分题正确率 %.1f%% (%d/%d)' % (100.0 * tot['ok'] / scored, tot['ok'], scored))
    if secs:
        print('   单题用时 avg %.1fs p95 %.1fs max %.1fs' %
              (statistics.mean(secs), sorted(secs)[int(0.95 * (len(secs) - 1))], max(secs)))
    if toks:
        print('   生成 token avg %d max %d 总 %d' % (int(statistics.mean(toks)), max(toks), sum(toks)))
    if outs:
        print('   prompt token avg %d max %d' % (int(statistics.mean(outs)), max(outs)))
    print('   %-14s %5s %5s %6s %6s' % ('domain', 'ok', 'bad', 'unsc', 'deg'))
    for k, v in sorted(stats.items()):
        print('   %-14s %5d %5d %6d %6d' % (k, v[0], v[1], v[2], v[3]))
    worst_rep.sort(reverse=True)
    print('   复读率最高的 5 条:')
    for deg, i, dom, got in worst_rep[:5]:
        print('     degen=%.2f i=%s %s %s' % (deg, i, dom, got))
    return tot


def main():
    grand = {'ok': 0, 'bad': 0, 'un': 0, 'deg': 0}
    for p in sys.argv[1:]:
        t = summarize(p)
        for k in grand:
            grand[k] += t[k]
        print()
    if len(sys.argv) > 2:
        scored = grand['ok'] + grand['bad']
        print('总计: 正确 %d 错误 %d 未判定 %d 崩坏 %d 正确率 %.1f%%' %
              (grand['ok'], grand['bad'], grand['un'], grand['deg'],
               100.0 * grand['ok'] / scored if scored else 0.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
